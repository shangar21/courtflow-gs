"""Rigid moving-least-squares deformation: batched PyTorch reference (autograd correctness oracle).

For Gaussian i with K control-point neighbours j (rest p_j, translation t_j, weights w_ij, rows
summing to 1):
    p'_j = p_j + t_j,  p* = sum w p,  q* = sum w p',  M = sum w (p - p*)(p' - q*)^T
    M = U S V^T,  R_i = V U^T with the reflection fix (last column of V flipped when det < 0)
    x'_i = R_i (x_i - p*) + q*
Optional per-control-point twist (blend): qbar_i = normalize(sum_j w_ij s_j qhat_j) with qhat_j the
unit control quaternion sign-aligned to the identity hemisphere, tau_i = normalize((1-beta) e + beta
qbar_i), Rf_i = R(tau_i) R_i. Orientation and band-1 SH use Rf_i; positions use R_i only.
Orientation: quat'_i = quaternion(Rf_i R(quat_i)), canonicalized to w >= 0.
Band-1 SH (gsplat order, colour = C1 (-y c0 + z c1 - x c2)): basis = C1 A d with
A = [[0,-1,0],[0,0,1],[-1,0,0]], so rotated coefficients are c' = A Rf A^T c."""
from __future__ import annotations
import torch
import torch.nn.functional as F

SH1_PERM = torch.tensor([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1); w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Shepperd's method (branch on the largest of trace / diagonal): differentiable per branch,
    stable for every proper rotation. Output canonicalized to w >= 0."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    cands = torch.stack((tr, m00, m11, m22), -1); branch = cands.argmax(-1)
    def safe_sqrt(x): return torch.sqrt(x.clamp_min(1e-12))
    s0 = safe_sqrt(1 + tr) * 2
    q0 = torch.stack((0.25*s0, (m[..., 2, 1]-m[..., 1, 2])/s0, (m[..., 0, 2]-m[..., 2, 0])/s0, (m[..., 1, 0]-m[..., 0, 1])/s0), -1)
    s1 = safe_sqrt(1 + m00 - m11 - m22) * 2
    q1 = torch.stack(((m[..., 2, 1]-m[..., 1, 2])/s1, 0.25*s1, (m[..., 0, 1]+m[..., 1, 0])/s1, (m[..., 0, 2]+m[..., 2, 0])/s1), -1)
    s2 = safe_sqrt(1 + m11 - m00 - m22) * 2
    q2 = torch.stack(((m[..., 0, 2]-m[..., 2, 0])/s2, (m[..., 0, 1]+m[..., 1, 0])/s2, 0.25*s2, (m[..., 1, 2]+m[..., 2, 1])/s2), -1)
    s3 = safe_sqrt(1 + m22 - m00 - m11) * 2
    q3 = torch.stack(((m[..., 1, 0]-m[..., 0, 1])/s3, (m[..., 0, 2]+m[..., 2, 0])/s3, (m[..., 1, 2]+m[..., 2, 1])/s3, 0.25*s3), -1)
    b = branch[..., None]
    q = torch.where(b == 0, q0, torch.where(b == 1, q1, torch.where(b == 2, q2, q3)))
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def blend_twist(q: torch.Tensor, nbr: torch.Tensor, w: torch.Tensor, beta: float) -> torch.Tensor:
    """Per-Gaussian twist quaternion tau (wxyz) from the neighbours' control rotations."""
    qh = F.normalize(q[nbr], dim=-1)
    qh = qh * torch.where(qh[..., :1] < 0, -1.0, 1.0)
    qbar = F.normalize((w[..., None] * qh).sum(-2), dim=-1)
    ident = torch.zeros_like(qbar); ident[..., 0] = 1
    return F.normalize((1 - beta) * ident + beta * qbar, dim=-1)


def mls_rotation(rest: torch.Tensor, nbr: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (R [N,3,3], p* [N,3], q* [N,3])."""
    p = rest[nbr]; pd = p + t[nbr]
    p_star = (w[..., None] * p).sum(-2); q_star = (w[..., None] * pd).sum(-2)
    M = torch.einsum("nk,nki,nkj->nij", w, p - p_star[:, None], pd - q_star[:, None])
    U, _, Vh = torch.linalg.svd(M)
    V = Vh.transpose(-2, -1)
    det = torch.det(V @ U.transpose(-2, -1))
    flip = torch.ones_like(V[..., 0, :]); flip[..., -1] = torch.where(det < 0, -1.0, 1.0)
    R = (V * flip[..., None, :]) @ U.transpose(-2, -1)
    return R, p_star, q_star


def rigid_mls(means: torch.Tensor, quats: torch.Tensor, sh1: torch.Tensor | None, rest: torch.Tensor, nbr: torch.Tensor, w: torch.Tensor,
              t: torch.Tensor, q: torch.Tensor | None = None, blend: bool = False, beta: float = 0.0, eps: float = 1e-6):
    """Reference deform. Returns (means' [N,3], quats' [N,4] wxyz with w >= 0, sh1' [N,3,3] or None).
    `eps` is unused here (autograd through torch.linalg.svd); it keeps the kernel's signature."""
    nbr = nbr.long()
    R, p_star, q_star = mls_rotation(rest, nbr, w, t)
    x = torch.einsum("nij,nj->ni", R, means - p_star) + q_star
    Rf = R
    if blend and q is not None and beta > 0:
        Rf = quat_to_matrix(blend_twist(q, nbr, w, beta)) @ R
    quat = matrix_to_quat(Rf @ quat_to_matrix(quats))
    sh = None
    if sh1 is not None:
        A = SH1_PERM.to(R)
        sh = (A @ Rf @ A.T) @ sh1
    return x, quat, sh
