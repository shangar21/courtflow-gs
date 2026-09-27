// Fused rigid-MLS deformation (forward + backward) for ring_init Stage B.
// Independent implementation (QuickCapture is not imported or linked).
//
// Per Gaussian i with K control neighbours (rest p_j, translation t_j, weights w_ij, sum w = 1):
//   p* = sum w p, q* = sum w (p + t), M = sum w (p - p*)(p + t - q*)^T = U S V^T,
//   R = V U^T (reflection-fixed), x' = R (x - p*) + q*.
// Optional twist from control rotations (blend): tau = normalize((1-b) e + b normalize(sum w s qhat)),
//   Rf = R(tau) R; orientation' = Rf * orientation, band-1 SH' = (A Rf A^T) SH.
//
// SVD: cyclic Jacobi on the symmetric (Frobenius-normalized) M^T M gives V (det +1) and sigma^2;
// U = [normalize(M v1), Gram-Schmidt(M v2), u1 x u2]. Choosing u3 = u1 x u2 yields exactly the
// reflection-fixed closest rotation Q = U V^T = R^T, with a *signed* third singular value
// sigma3 = u3 . M v3 (negative in the reflection case).
// Backward: for Q = polar(M), dL/dM = U Y V^T, Y_ij = (X_ij - X_ji) / (s_i + s_j), X = U^T G_Q V
// (s signed). This is the derivative of the orthogonal polar factor; it only divides by s_i + s_j,
// which is far better conditioned than the 1/(s_j^2 - s_i^2) terms of a full SVD gradient (near-planar
// neighbourhoods with s3 ~ 0 stay finite). |s_i + s_j| is clamped to eps * s1 (eps = config.mls_svd_epsilon).
// Rotation-valued outputs (orientation, SH) back-propagate through their tangent space: a left
// perturbation Rf -> exp([psi]x) Rf, converted to a matrix gradient [psi/2]x R on R.
//
// Atomics: the (Gaussian, neighbour-slot) gradient for control point j is reduced across the warp with a
// segmented inclusive scan (__shfl_up_sync) over runs of equal j before one atomicAdd per run. The
// reduction is exact for any order; it is most effective when Gaussians are sorted by nearest control
// point (Stage B sorts them), so adjacent lanes share neighbours.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define FULL_MASK 0xffffffffu

namespace {

__device__ __forceinline__ void mat_zero(float* a) {
#pragma unroll
  for (int i = 0; i < 9; ++i) a[i] = 0.f;
}

__device__ __forceinline__ void quat_to_mat(const float* q, float* R) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  float n = rsqrtf(w * w + x * x + y * y + z * z);
  w *= n; x *= n; y *= n; z *= n;
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w); R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w); R[7] = 2 * (y * z + x * w); R[8] = 1 - 2 * (x * x + y * y);
}

// Shepperd's method; R must be a proper rotation.
__device__ __forceinline__ void mat_to_quat(const float* m, float* q) {
  float tr = m[0] + m[4] + m[8];
  if (tr >= m[0] && tr >= m[4] && tr >= m[8]) {
    float s = sqrtf(fmaxf(1.f + tr, 1e-12f)) * 2.f;
    q[0] = 0.25f * s; q[1] = (m[7] - m[5]) / s; q[2] = (m[2] - m[6]) / s; q[3] = (m[3] - m[1]) / s;
  } else if (m[0] >= m[4] && m[0] >= m[8]) {
    float s = sqrtf(fmaxf(1.f + m[0] - m[4] - m[8], 1e-12f)) * 2.f;
    q[0] = (m[7] - m[5]) / s; q[1] = 0.25f * s; q[2] = (m[1] + m[3]) / s; q[3] = (m[2] + m[6]) / s;
  } else if (m[4] >= m[8]) {
    float s = sqrtf(fmaxf(1.f + m[4] - m[0] - m[8], 1e-12f)) * 2.f;
    q[0] = (m[2] - m[6]) / s; q[1] = (m[1] + m[3]) / s; q[2] = 0.25f * s; q[3] = (m[5] + m[7]) / s;
  } else {
    float s = sqrtf(fmaxf(1.f + m[8] - m[0] - m[4], 1e-12f)) * 2.f;
    q[0] = (m[3] - m[1]) / s; q[1] = (m[2] + m[6]) / s; q[2] = (m[5] + m[7]) / s; q[3] = 0.25f * s;
  }
}

__device__ __forceinline__ void quat_mul(const float* a, const float* b, float* o) {
  o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

__device__ __forceinline__ void normalize4(float* q) {
  float n = rsqrtf(fmaxf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1e-30f));
  q[0] *= n; q[1] *= n; q[2] *= n; q[3] *= n;
}

__device__ __forceinline__ void matmul(const float* a, const float* b, float* c) {  // c = a b
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}

__device__ __forceinline__ void matmul_tn(const float* a, const float* b, float* c) {  // c = a^T b
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) c[i * 3 + j] = a[i] * b[j] + a[3 + i] * b[3 + j] + a[6 + i] * b[6 + j];
}

__device__ __forceinline__ void matmul_nt(const float* a, const float* b, float* c) {  // c = a b^T
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) c[i * 3 + j] = a[i * 3] * b[j * 3] + a[i * 3 + 1] * b[j * 3 + 1] + a[i * 3 + 2] * b[j * 3 + 2];
}

__device__ __forceinline__ void cross3(const float* a, const float* b, float* c) {
  c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0];
}

// Symmetric 3x3 eigen-decomposition by cyclic Jacobi. On return S is (nearly) diagonal and the
// columns of V are the eigenvectors.
__device__ void jacobi_eigen3(float* S, float* V) {
  V[0] = 1; V[1] = 0; V[2] = 0; V[3] = 0; V[4] = 1; V[5] = 0; V[6] = 0; V[7] = 0; V[8] = 1;
  const int P[3] = {0, 0, 1}, Q[3] = {1, 2, 2};
  for (int sweep = 0; sweep < 8; ++sweep) {
    float off = S[1] * S[1] + S[2] * S[2] + S[5] * S[5];
    float diag = S[0] * S[0] + S[4] * S[4] + S[8] * S[8];
    if (off <= 1e-14f * diag) break;
#pragma unroll
    for (int r = 0; r < 3; ++r) {
      int p = P[r], q = Q[r];
      float apq = S[p * 3 + q];
      if (fabsf(apq) < 1e-30f) continue;
      float app = S[p * 3 + p], aqq = S[q * 3 + q];
      float theta = (aqq - app) / (2.f * apq);
      float t = copysignf(1.f, theta) / (fabsf(theta) + sqrtf(theta * theta + 1.f));
      float c = rsqrtf(t * t + 1.f), s = t * c;
      // S <- J^T S J with J the (p,q) rotation.
#pragma unroll
      for (int k = 0; k < 3; ++k) {  // columns
        float skp = S[k * 3 + p], skq = S[k * 3 + q];
        S[k * 3 + p] = c * skp - s * skq; S[k * 3 + q] = s * skp + c * skq;
      }
#pragma unroll
      for (int k = 0; k < 3; ++k) {  // rows
        float spk = S[p * 3 + k], sqk = S[q * 3 + k];
        S[p * 3 + k] = c * spk - s * sqk; S[q * 3 + k] = s * spk + c * sqk;
      }
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
        V[k * 3 + p] = c * vkp - s * vkq; V[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
}

// Reflection-fixed polar factor of M: returns U, V (columns), signed singular values s, and
// R = V U^T (so that R^T is the closest proper rotation to M).
__device__ void polar_svd(const float* M, float* U, float* V, float* s, float* R) {
  float fro = sqrtf(fmaxf(M[0] * M[0] + M[1] * M[1] + M[2] * M[2] + M[3] * M[3] + M[4] * M[4] + M[5] * M[5] + M[6] * M[6] + M[7] * M[7] + M[8] * M[8], 1e-37f));
  float Mn[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) Mn[i] = M[i] / fro;
  float S[9];
  matmul_tn(Mn, Mn, S);
  jacobi_eigen3(S, V);
  // Sort eigenpairs descending.
  float ev[3] = {S[0], S[4], S[8]};
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2 - a; ++b)
      if (ev[b] < ev[b + 1]) {
        float tmp = ev[b]; ev[b] = ev[b + 1]; ev[b + 1] = tmp;
#pragma unroll
        for (int k = 0; k < 3; ++k) { float tv = V[k * 3 + b]; V[k * 3 + b] = V[k * 3 + b + 1]; V[k * 3 + b + 1] = tv; }
      }
  // Proper V: v3 = v1 x v2.
  float v1[3] = {V[0], V[3], V[6]}, v2[3] = {V[1], V[4], V[7]}, v3[3];
  cross3(v1, v2, v3);
  V[2] = v3[0]; V[5] = v3[1]; V[8] = v3[2];
  float mv1[3], mv2[3], mv3[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    mv1[i] = Mn[i * 3] * v1[0] + Mn[i * 3 + 1] * v1[1] + Mn[i * 3 + 2] * v1[2];
    mv2[i] = Mn[i * 3] * v2[0] + Mn[i * 3 + 1] * v2[1] + Mn[i * 3 + 2] * v2[2];
    mv3[i] = Mn[i * 3] * v3[0] + Mn[i * 3 + 1] * v3[1] + Mn[i * 3 + 2] * v3[2];
  }
  float u1[3], u2[3], u3[3];
  float n1 = sqrtf(mv1[0] * mv1[0] + mv1[1] * mv1[1] + mv1[2] * mv1[2]);
  if (n1 > 1e-20f) { u1[0] = mv1[0] / n1; u1[1] = mv1[1] / n1; u1[2] = mv1[2] / n1; }
  else { u1[0] = 1; u1[1] = 0; u1[2] = 0; }
  float d12 = u1[0] * mv2[0] + u1[1] * mv2[1] + u1[2] * mv2[2];
  u2[0] = mv2[0] - d12 * u1[0]; u2[1] = mv2[1] - d12 * u1[1]; u2[2] = mv2[2] - d12 * u1[2];
  float n2 = sqrtf(u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
  if (n2 > 1e-7f * fmaxf(n1, 1e-20f)) { u2[0] /= n2; u2[1] /= n2; u2[2] /= n2; }
  else {  // collinear neighbourhood: any unit vector orthogonal to u1
    float a[3] = {fabsf(u1[0]) < 0.9f ? 1.f : 0.f, fabsf(u1[0]) < 0.9f ? 0.f : 1.f, 0.f};
    float d = a[0] * u1[0] + a[1] * u1[1];
    u2[0] = a[0] - d * u1[0]; u2[1] = a[1] - d * u1[1]; u2[2] = -d * u1[2];
    float nn = rsqrtf(u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
    u2[0] *= nn; u2[1] *= nn; u2[2] *= nn;
  }
  cross3(u1, u2, u3);
  U[0] = u1[0]; U[3] = u1[1]; U[6] = u1[2];
  U[1] = u2[0]; U[4] = u2[1]; U[7] = u2[2];
  U[2] = u3[0]; U[5] = u3[1]; U[8] = u3[2];
  s[0] = fro * (u1[0] * mv1[0] + u1[1] * mv1[1] + u1[2] * mv1[2]);
  s[1] = fro * (u2[0] * mv2[0] + u2[1] * mv2[1] + u2[2] * mv2[2]);
  s[2] = fro * (u3[0] * mv3[0] + u3[1] * mv3[1] + u3[2] * mv3[2]);  // signed
  matmul_nt(V, U, R);  // R = V U^T
}

// Band-1 SH basis permutation A (gsplat: colour = C1 (-y c0 + z c1 - x c2) = C1 (A d) . c).
__device__ __forceinline__ void sh_rotation(const float* Rf, float* B) {  // B = A Rf A^T
  // A = [[0,-1,0],[0,0,1],[-1,0,0]] => (A X A^T)_ij = sum_ab A_ia X_ab A_jb with A rows:
  // row0 -> -e1, row1 -> +e2, row2 -> -e0.
  const int ai[3] = {1, 2, 0};
  const float as[3] = {-1.f, 1.f, -1.f};
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) B[i * 3 + j] = as[i] * as[j] * Rf[ai[i] * 3 + ai[j]];
}

__device__ __forceinline__ void sh_rotation_adjoint(const float* GB, float* GR) {  // GR = A^T GB A
  const int ai[3] = {1, 2, 0};
  const float as[3] = {-1.f, 1.f, -1.f};
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) GR[ai[i] * 3 + ai[j]] = as[i] * as[j] * GB[i * 3 + j];
}

template <int G>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int off = G / 2; off > 0; off >>= 1) v += __shfl_xor_sync(FULL_MASK, v, off, G);
  return v;
}

// Warp-level segmented reduction of `nc` components keyed by control index, one atomicAdd per run.
__device__ __forceinline__ void warp_segment_atomic(float* base, int key, const float* v, int nc, bool valid) {
  const int lane = threadIdx.x & 31;
  int k = valid ? key : -1;
  int prev = __shfl_up_sync(FULL_MASK, k, 1);
  int next = __shfl_down_sync(FULL_MASK, k, 1);
  bool head = (lane == 0) || (prev != k);
  bool tail = (lane == 31) || (next != k);
  unsigned heads = __ballot_sync(FULL_MASK, head);
  int seg_start = 31 - __clz(heads & (FULL_MASK >> (31 - lane)));
  for (int c = 0; c < nc; ++c) {
    float x = valid ? v[c] : 0.f;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      float o = __shfl_up_sync(FULL_MASK, x, d);
      if (lane - d >= seg_start) x += o;
    }
    if (tail && valid) atomicAdd(base + (size_t)key * nc + c, x);
  }
}

// Blend twist: tau from control quaternions; also returns m (unnormalized mean) norms for backward.
__device__ __forceinline__ void load_unit_aligned(const float* q, float* qh, float& inv_norm, float& sgn) {
  float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  inv_norm = rsqrtf(fmaxf(n2, 1e-30f));
  sgn = (q[0] * inv_norm < 0.f) ? -1.f : 1.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) qh[c] = q[c] * inv_norm;
}

template <int G>
__global__ void mls_forward_kernel(
    int N, int K, const float* __restrict__ means, const float* __restrict__ quats, const float* __restrict__ sh1,
    const float* __restrict__ rest, const int* __restrict__ nbr, const float* __restrict__ w, const float* __restrict__ t,
    const float* __restrict__ cq, bool blend, float beta,
    float* __restrict__ out_means, float* __restrict__ out_quats, float* __restrict__ out_sh1,
    float* __restrict__ sR, float* __restrict__ sU, float* __restrict__ sV, float* __restrict__ ssig,
    float* __restrict__ spstar, float* __restrict__ stau) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gid = tid / G, sub = tid % G;
  const bool valid = gid < N;
  const int gi = valid ? gid : 0;
  float ps[3] = {0, 0, 0}, qs[3] = {0, 0, 0}, m4[4] = {0, 0, 0, 0};
  for (int k = sub; k < K; k += G) {
    if (!valid) break;
    int j = nbr[gi * K + k]; float wk = w[gi * K + k];
#pragma unroll
    for (int c = 0; c < 3; ++c) { float p = rest[j * 3 + c]; ps[c] += wk * p; qs[c] += wk * (p + t[j * 3 + c]); }
    if (blend) {
      float qh[4], inv, sg; load_unit_aligned(cq + j * 4, qh, inv, sg);
#pragma unroll
      for (int c = 0; c < 4; ++c) m4[c] += wk * sg * qh[c];
    }
  }
#pragma unroll
  for (int c = 0; c < 3; ++c) { ps[c] = group_sum<G>(ps[c]); qs[c] = group_sum<G>(qs[c]); }
  if (blend) {
#pragma unroll
    for (int c = 0; c < 4; ++c) m4[c] = group_sum<G>(m4[c]);
  }
  float M[9]; mat_zero(M);
  for (int k = sub; k < K; k += G) {
    if (!valid) break;
    int j = nbr[gi * K + k]; float wk = w[gi * K + k];
    float a[3], b[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) { float p = rest[j * 3 + c]; a[c] = wk * (p - ps[c]); b[c] = p + t[j * 3 + c] - qs[c]; }
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
      for (int c = 0; c < 3; ++c) M[r * 3 + c] += a[r] * b[c];
  }
#pragma unroll
  for (int i = 0; i < 9; ++i) M[i] = group_sum<G>(M[i]);
  if (!valid || sub != 0) return;

  float U[9], V[9], s[3], R[9];
  polar_svd(M, U, V, s, R);
  float x[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) x[c] = means[gi * 3 + c] - ps[c];
#pragma unroll
  for (int r = 0; r < 3; ++r) out_means[gi * 3 + r] = R[r * 3] * x[0] + R[r * 3 + 1] * x[1] + R[r * 3 + 2] * x[2] + qs[r];
  float tau[4] = {1, 0, 0, 0};
  if (blend) {
    float qbar[4] = {m4[0], m4[1], m4[2], m4[3]}; normalize4(qbar);
#pragma unroll
    for (int c = 0; c < 4; ++c) tau[c] = beta * qbar[c] + (c == 0 ? 1.f - beta : 0.f);
    normalize4(tau);
  }
  float qR[4], qRf[4], qrot[4] = {quats[gi * 4], quats[gi * 4 + 1], quats[gi * 4 + 2], quats[gi * 4 + 3]}, o[4];
  mat_to_quat(R, qR); normalize4(qrot);
  quat_mul(tau, qR, qRf); quat_mul(qRf, qrot, o); normalize4(o);
  float sg = o[0] < 0.f ? -1.f : 1.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) out_quats[gi * 4 + c] = sg * o[c];
  if (sh1 != nullptr) {
    float T[9], Rf[9], B[9];
    quat_to_mat(tau, T); matmul(T, R, Rf); sh_rotation(Rf, B);
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
      for (int c = 0; c < 3; ++c)
        out_sh1[gi * 9 + a * 3 + c] = B[a * 3] * sh1[gi * 9 + c] + B[a * 3 + 1] * sh1[gi * 9 + 3 + c] + B[a * 3 + 2] * sh1[gi * 9 + 6 + c];
  }
#pragma unroll
  for (int i = 0; i < 9; ++i) { sR[gi * 9 + i] = R[i]; sU[gi * 9 + i] = U[i]; sV[gi * 9 + i] = V[i]; }
#pragma unroll
  for (int c = 0; c < 3; ++c) { ssig[gi * 3 + c] = s[c]; spstar[gi * 3 + c] = ps[c]; }
#pragma unroll
  for (int c = 0; c < 4; ++c) stau[gi * 4 + c] = tau[c];
}

template <int G>
__global__ void mls_backward_kernel(
    int N, int K, const float* __restrict__ g_means, const float* __restrict__ g_quats, const float* __restrict__ g_sh1,
    const float* __restrict__ g_R, const float* __restrict__ means, const float* __restrict__ sh1, const float* __restrict__ out_quats,
    const float* __restrict__ rest, const int* __restrict__ nbr, const float* __restrict__ w, const float* __restrict__ cq,
    const float* __restrict__ sR, const float* __restrict__ sU, const float* __restrict__ sV, const float* __restrict__ ssig,
    const float* __restrict__ spstar, const float* __restrict__ stau, bool blend, float beta, float eps,
    float* __restrict__ grad_t, float* __restrict__ grad_q) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gid = tid / G, sub = tid % G;
  const bool valid = gid < N;
  const int gi = valid ? gid : 0;

  // Per-Gaussian dL/dM and dL/dq* (computed redundantly by the G lanes of a group).
  float R[9], U[9], V[9], s[3], ps[3], tau[4];
#pragma unroll
  for (int i = 0; i < 9; ++i) { R[i] = sR[gi * 9 + i]; U[i] = sU[gi * 9 + i]; V[i] = sV[gi * 9 + i]; }
#pragma unroll
  for (int c = 0; c < 3; ++c) { s[c] = ssig[gi * 3 + c]; ps[c] = spstar[gi * 3 + c]; }
#pragma unroll
  for (int c = 0; c < 4; ++c) tau[c] = stau[gi * 4 + c];
  float gx[3] = {0, 0, 0}, go[4] = {0, 0, 0, 0}, o[4];
  if (g_means != nullptr) {
#pragma unroll
    for (int c = 0; c < 3; ++c) gx[c] = g_means[gi * 3 + c];
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) { o[c] = out_quats[gi * 4 + c]; if (g_quats != nullptr) go[c] = g_quats[gi * 4 + c]; }
  // Orientation: left perturbation psi of Rf.  dL/dpsi = 0.5 (-gw ov + ow gv + ov x gv).
  float gpsi[3];
  {
    float ov[3] = {o[1], o[2], o[3]}, gv[3] = {go[1], go[2], go[3]}, cr[3];
    cross3(ov, gv, cr);
#pragma unroll
    for (int c = 0; c < 3; ++c) gpsi[c] = 0.5f * (-go[0] * ov[c] + o[0] * gv[c] + cr[c]);
  }
  float T[9];
  quat_to_mat(tau, T);
  if (sh1 != nullptr && g_sh1 != nullptr) {
    float Rf[9], GB[9], GR[9], H[9];
    matmul(T, R, Rf);
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
      for (int b = 0; b < 3; ++b)
        GB[a * 3 + b] = g_sh1[gi * 9 + a * 3] * sh1[gi * 9 + b * 3] + g_sh1[gi * 9 + a * 3 + 1] * sh1[gi * 9 + b * 3 + 1] + g_sh1[gi * 9 + a * 3 + 2] * sh1[gi * 9 + b * 3 + 2];
    sh_rotation_adjoint(GB, GR);
    matmul_nt(GR, Rf, H);  // H = G_Rf Rf^T
    gpsi[0] += H[7] - H[5]; gpsi[1] += H[2] - H[6]; gpsi[2] += H[3] - H[1];
  }
  // psi on Rf = T R  ->  omega on R: T^T psi.
  float gom[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) gom[c] = T[c] * gpsi[0] + T[3 + c] * gpsi[1] + T[6 + c] * gpsi[2];
  // Matrix gradient on R: position term gx (x - p*)^T plus rotation term [gom/2]x R.
  float GRm[9];
  {
    float xm[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) xm[c] = means[gi * 3 + c] - ps[c];
    float h[3] = {0.5f * gom[0], 0.5f * gom[1], 0.5f * gom[2]};
    float Hx[9] = {0, -h[2], h[1], h[2], 0, -h[0], -h[1], h[0], 0}, HR[9];
    matmul(Hx, R, HR);
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
      for (int c = 0; c < 3; ++c) GRm[r * 3 + c] = gx[r] * xm[c] + HR[r * 3 + c] + (g_R != nullptr ? g_R[gi * 9 + r * 3 + c] : 0.f);
  }
  // Polar backward: Q = R^T = U V^T, G_Q = G_R^T; X = U^T G_Q V; Y = skew(X)/(s_i+s_j); dL/dM = U Y V^T.
  float Gm[9];
  {
    float GQ[9], tmp[9], X[9], Y[9];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
      for (int c = 0; c < 3; ++c) GQ[r * 3 + c] = GRm[c * 3 + r];
    matmul_tn(U, GQ, tmp); matmul(tmp, V, X);
    float floor_ = eps * fmaxf(fabsf(s[0]), 1e-30f);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        if (i == j) { Y[i * 3 + j] = 0.f; continue; }
        float den = s[i] + s[j];
        if (fabsf(den) < floor_) den = copysignf(floor_, den);
        Y[i * 3 + j] = (X[i * 3 + j] - X[j * 3 + i]) / den;
      }
    matmul(U, Y, tmp); matmul_nt(tmp, V, Gm);
  }
  // Blend: gradient on tau from gpsi (left perturbation): g_tau = 2 (0, gpsi) (x) tau.
  float gm4[4] = {0, 0, 0, 0};
  float sum_a[3] = {0, 0, 0};
  float m4[4] = {0, 0, 0, 0};
  for (int k = sub; k < K; k += G) {
    if (!valid) break;
    int j = nbr[gi * K + k]; float wk = w[gi * K + k];
#pragma unroll
    for (int c = 0; c < 3; ++c) sum_a[c] += wk * (rest[j * 3 + c] - ps[c]);
    if (blend) {
      float qh[4], inv, sg; load_unit_aligned(cq + j * 4, qh, inv, sg);
#pragma unroll
      for (int c = 0; c < 4; ++c) m4[c] += wk * sg * qh[c];
    }
  }
#pragma unroll
  for (int c = 0; c < 3; ++c) sum_a[c] = group_sum<G>(sum_a[c]);
  if (blend) {
#pragma unroll
    for (int c = 0; c < 4; ++c) m4[c] = group_sum<G>(m4[c]);
    float gtau[4];
    {
      float tv[3] = {tau[1], tau[2], tau[3]}, cr[3];
      cross3(gpsi, tv, cr);
      gtau[0] = -2.f * (gpsi[0] * tv[0] + gpsi[1] * tv[1] + gpsi[2] * tv[2]);
#pragma unroll
      for (int c = 0; c < 3; ++c) gtau[1 + c] = 2.f * (tau[0] * gpsi[c] + cr[c]);
    }
    // tau = n/|n|, n = (1-b) e + b qbar, qbar = m/|m|.
    float qbar[4] = {m4[0], m4[1], m4[2], m4[3]};
    float mnorm = sqrtf(fmaxf(qbar[0] * qbar[0] + qbar[1] * qbar[1] + qbar[2] * qbar[2] + qbar[3] * qbar[3], 1e-30f));
#pragma unroll
    for (int c = 0; c < 4; ++c) qbar[c] /= mnorm;
    float n[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) n[c] = beta * qbar[c] + (c == 0 ? 1.f - beta : 0.f);
    float nnorm = sqrtf(fmaxf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3], 1e-30f));
    float d1 = gtau[0] * tau[0] + gtau[1] * tau[1] + gtau[2] * tau[2] + gtau[3] * tau[3];
    float gqbar[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) gqbar[c] = beta * (gtau[c] - d1 * tau[c]) / nnorm;
    float d2 = gqbar[0] * qbar[0] + gqbar[1] * qbar[1] + gqbar[2] * qbar[2] + gqbar[3] * qbar[3];
#pragma unroll
    for (int c = 0; c < 4; ++c) gm4[c] = (gqbar[c] - d2 * qbar[c]) / mnorm;
  }
  // dL/dq* from M (sum a_k ~ 0 but kept exact) plus position.
  float gqs[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) gqs[c] = gx[c] - (Gm[c] * sum_a[0] + Gm[3 + c] * sum_a[1] + Gm[6 + c] * sum_a[2]);
  // Per neighbour: grad t_j = Gm^T a_k + w_k gqs; grad q_j via the twist.
  const int iters = (K + G - 1) / G;
  for (int it = 0; it < iters; ++it) {
    int k = sub + it * G;
    bool live = valid && k < K;
    int j = live ? nbr[gi * K + k] : 0;
    float gt[3] = {0, 0, 0}, gq[4] = {0, 0, 0, 0};
    if (live) {
      float wk = w[gi * K + k], a[3];
#pragma unroll
      for (int c = 0; c < 3; ++c) a[c] = wk * (rest[j * 3 + c] - ps[c]);
#pragma unroll
      for (int c = 0; c < 3; ++c) gt[c] = Gm[c] * a[0] + Gm[3 + c] * a[1] + Gm[6 + c] * a[2] + wk * gqs[c];
      if (blend) {
        float qh[4], inv, sg; load_unit_aligned(cq + j * 4, qh, inv, sg);
        float gqh[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) gqh[c] = wk * sg * gm4[c];
        float d = gqh[0] * qh[0] + gqh[1] * qh[1] + gqh[2] * qh[2] + gqh[3] * qh[3];
#pragma unroll
        for (int c = 0; c < 4; ++c) gq[c] = (gqh[c] - d * qh[c]) * inv;
      }
    }
    warp_segment_atomic(grad_t, j, gt, 3, live);
    if (blend) warp_segment_atomic(grad_q, j, gq, 4, live);
  }
}

template <int G>
void launch_forward(int N, int K, const float* means, const float* quats, const float* sh1, const float* rest, const int* nbr,
                    const float* w, const float* t, const float* cq, bool blend, float beta, float* om, float* oq, float* osh,
                    float* R, float* U, float* V, float* sig, float* ps, float* tau, cudaStream_t stream) {
  const int threads = 256;
  const long total = (long)N * G;
  const int blocks = (int)((total + threads - 1) / threads);
  mls_forward_kernel<G><<<blocks, threads, 0, stream>>>(N, K, means, quats, sh1, rest, nbr, w, t, cq, blend, beta, om, oq, osh, R, U, V, sig, ps, tau);
}

template <int G>
void launch_backward(int N, int K, const float* gm, const float* gq, const float* gs, const float* gR, const float* means, const float* sh1, const float* oq,
                     const float* rest, const int* nbr, const float* w, const float* cq, const float* R, const float* U, const float* V,
                     const float* sig, const float* ps, const float* tau, bool blend, float beta, float eps, float* grad_t, float* grad_q,
                     cudaStream_t stream) {
  const int threads = 256;
  const long total = (long)N * G;
  const int blocks = (int)((total + threads - 1) / threads);
  mls_backward_kernel<G><<<blocks, threads, 0, stream>>>(N, K, gm, gq, gs, gR, means, sh1, oq, rest, nbr, w, cq, R, U, V, sig, ps, tau, blend, beta, eps, grad_t, grad_q);
}

}  // namespace

#define CHECK_F32(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == at::kFloat, #x " must be a contiguous float32 CUDA tensor")

std::vector<torch::Tensor> mls_forward(torch::Tensor means, torch::Tensor quats, torch::Tensor sh1, torch::Tensor rest, torch::Tensor nbr,
                                       torch::Tensor w, torch::Tensor t, torch::Tensor cq, bool blend, double beta, int64_t group) {
  CHECK_F32(means); CHECK_F32(quats); CHECK_F32(rest); CHECK_F32(w); CHECK_F32(t);
  TORCH_CHECK(nbr.is_cuda() && nbr.is_contiguous() && nbr.scalar_type() == at::kInt, "nbr must be contiguous int32 CUDA");
  const int N = means.size(0), K = nbr.size(1);
  const bool has_sh = sh1.numel() > 0;
  if (has_sh) CHECK_F32(sh1);
  blend = blend && cq.numel() > 0;
  if (blend) CHECK_F32(cq);
  auto opts = means.options();
  auto om = torch::empty({N, 3}, opts), oq = torch::empty({N, 4}, opts);
  auto osh = has_sh ? torch::empty({N, 3, 3}, opts) : torch::empty({0}, opts);
  auto R = torch::empty({N, 9}, opts), U = torch::empty({N, 9}, opts), V = torch::empty({N, 9}, opts);
  auto sig = torch::empty({N, 3}, opts), ps = torch::empty({N, 3}, opts), tau = torch::empty({N, 4}, opts);
  if (N == 0) return {om, oq, osh, R, U, V, sig, ps, tau};
  auto stream = c10::cuda::getCurrentCUDAStream();
  const float* shp = has_sh ? sh1.data_ptr<float>() : nullptr;
  const float* cqp = blend ? cq.data_ptr<float>() : nullptr;
  float* oshp = has_sh ? osh.data_ptr<float>() : nullptr;
#define FWD(GG) launch_forward<GG>(N, K, means.data_ptr<float>(), quats.data_ptr<float>(), shp, rest.data_ptr<float>(), nbr.data_ptr<int>(), \
    w.data_ptr<float>(), t.data_ptr<float>(), cqp, blend, (float)beta, om.data_ptr<float>(), oq.data_ptr<float>(), oshp, R.data_ptr<float>(), \
    U.data_ptr<float>(), V.data_ptr<float>(), sig.data_ptr<float>(), ps.data_ptr<float>(), tau.data_ptr<float>(), stream)
  switch (group) {
    case 1: FWD(1); break;
    case 2: FWD(2); break;
    case 4: FWD(4); break;
    case 8: FWD(8); break;
    case 16: FWD(16); break;
    default: TORCH_CHECK(false, "group must be 1, 2, 4, 8 or 16");
  }
#undef FWD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {om, oq, osh, R, U, V, sig, ps, tau};
}

std::vector<torch::Tensor> mls_backward_r(torch::Tensor g_means, torch::Tensor g_quats, torch::Tensor g_sh1, torch::Tensor g_R, torch::Tensor means, torch::Tensor sh1,
                                        torch::Tensor out_quats, torch::Tensor rest, torch::Tensor nbr, torch::Tensor w, torch::Tensor cq,
                                        torch::Tensor R, torch::Tensor U, torch::Tensor V, torch::Tensor sig, torch::Tensor ps, torch::Tensor tau,
                                        bool blend, double beta, double eps, int64_t num_controls, int64_t group) {
  const int N = means.size(0), K = nbr.size(1);
  blend = blend && cq.numel() > 0;
  auto opts = means.options();
  auto grad_t = torch::zeros({num_controls, 3}, opts);
  auto grad_q = blend ? torch::zeros({num_controls, 4}, opts) : torch::empty({0}, opts);
  if (N == 0) return {grad_t, grad_q};
  const float* gm = g_means.numel() ? g_means.data_ptr<float>() : nullptr;
  const float* gq = g_quats.numel() ? g_quats.data_ptr<float>() : nullptr;
  const float* gs = g_sh1.numel() ? g_sh1.data_ptr<float>() : nullptr;
  if (g_R.numel()) CHECK_F32(g_R);
  const float* gR = g_R.numel() ? g_R.data_ptr<float>() : nullptr;
  const float* shp = sh1.numel() ? sh1.data_ptr<float>() : nullptr;
  const float* cqp = blend ? cq.data_ptr<float>() : nullptr;
  float* gqp = blend ? grad_q.data_ptr<float>() : nullptr;
  auto stream = c10::cuda::getCurrentCUDAStream();
#define BWD(GG) launch_backward<GG>(N, K, gm, gq, gs, gR, means.data_ptr<float>(), shp, out_quats.data_ptr<float>(), rest.data_ptr<float>(), \
    nbr.data_ptr<int>(), w.data_ptr<float>(), cqp, R.data_ptr<float>(), U.data_ptr<float>(), V.data_ptr<float>(), sig.data_ptr<float>(), \
    ps.data_ptr<float>(), tau.data_ptr<float>(), blend, (float)beta, (float)eps, grad_t.data_ptr<float>(), gqp, stream)
  switch (group) {
    case 1: BWD(1); break;
    case 2: BWD(2); break;
    case 4: BWD(4); break;
    case 8: BWD(8); break;
    case 16: BWD(16); break;
    default: TORCH_CHECK(false, "group must be 1, 2, 4, 8 or 16");
  }
#undef BWD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_t, grad_q};
}

// Backward without a rotation-output gradient (original API).
std::vector<torch::Tensor> mls_backward(torch::Tensor g_means, torch::Tensor g_quats, torch::Tensor g_sh1, torch::Tensor means, torch::Tensor sh1,
                                        torch::Tensor out_quats, torch::Tensor rest, torch::Tensor nbr, torch::Tensor w, torch::Tensor cq,
                                        torch::Tensor R, torch::Tensor U, torch::Tensor V, torch::Tensor sig, torch::Tensor ps, torch::Tensor tau,
                                        bool blend, double beta, double eps, int64_t num_controls, int64_t group) {
  return mls_backward_r(g_means, g_quats, g_sh1, torch::empty({0}, means.options()), means, sh1, out_quats, rest, nbr, w, cq, R, U, V, sig, ps, tau,
                        blend, beta, eps, num_controls, group);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mls_forward, "fused rigid MLS forward");
  m.def("backward", &mls_backward, "fused rigid MLS backward");
  m.def("backward_r", &mls_backward_r, "fused rigid MLS backward with an extra dL/dR [N,9] input");
}
