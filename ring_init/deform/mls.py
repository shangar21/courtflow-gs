"""Rigid-MLS dispatcher: fused CUDA kernel when available (float32, CUDA), else the PyTorch
reference (`mls_ref.rigid_mls`). Set RING_MLS_FORCE_REFERENCE=1 to force the reference.

    deform(means, quats, sh1, rest, nbr, w, t, q=None, blend=False, beta=0., eps=1e-6)
        -> (means' [N,3], quats' [N,4] wxyz (w >= 0), sh1' [N,3,3] or None)

Gradients flow to `t` (and `q` when blend is on); canonical inputs are treated as constants.
The kernel is JIT-built with torch.utils.cpp_extension.load (cached under csrc/build) or installed
with `python ring_init/deform/setup.py install` (module name `ring_mls_cuda`)."""
from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path
import torch
from ring_init.deform.mls_ref import rigid_mls

_CSRC = Path(__file__).parent / "csrc"
_EXT = None
_EXT_ERROR: Exception | None = None
DEFAULT_GROUP = int(os.environ.get("RING_MLS_GROUP", "1"))


def _nvcc_version(cuda_home: str) -> str | None:
    try:
        out = subprocess.run([str(Path(cuda_home) / "bin" / "nvcc"), "--version"], capture_output=True, text=True, check=True).stdout
        m = re.search(r"release (\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _matching_cuda_home() -> str | None:
    """A CUDA toolkit whose major version matches torch's (JIT builds refuse a mismatch)."""
    import torch.utils.cpp_extension as ce
    want = torch.version.cuda
    if want is None: return None
    candidates = [os.environ.get("CUDA_HOME"), ce.CUDA_HOME, f"/usr/local/cuda-{want}"] + sorted(str(p) for p in Path("/usr/local").glob(f"cuda-{want.split('.')[0]}*"))
    for home in candidates:
        if home and _nvcc_version(home) and _nvcc_version(home).split(".")[0] == want.split(".")[0]:
            return home
    return None


def load_extension(verbose: bool = False):
    """Import the installed extension or JIT-build it. Returns the module or raises."""
    global _EXT, _EXT_ERROR
    if _EXT is not None: return _EXT
    if _EXT_ERROR is not None: raise _EXT_ERROR
    try:
        import ring_mls_cuda  # installed via setup.py
        _EXT = ring_mls_cuda; return _EXT
    except ImportError:
        pass
    try:
        import torch.utils.cpp_extension as ce
        home = _matching_cuda_home()
        if home is None: raise RuntimeError(f"No CUDA toolkit matching torch CUDA {torch.version.cuda} found for the JIT build")
        ce.CUDA_HOME = home; os.environ["CUDA_HOME"] = home
        if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability(); os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        build = _CSRC / "build"; build.mkdir(parents=True, exist_ok=True)
        _EXT = ce.load(name="ring_mls_cuda", sources=[str(_CSRC / "mls_kernel.cu")], build_directory=str(build),
                       extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=verbose)
        return _EXT
    except Exception as error:  # remember the failure; callers fall back to the reference
        _EXT_ERROR = error
        raise


def kernel_available() -> bool:
    if os.environ.get("RING_MLS_FORCE_REFERENCE") == "1" or not torch.cuda.is_available(): return False
    try:
        load_extension(); return True
    except Exception:
        return False


_EMPTY: dict[torch.device, torch.Tensor] = {}


def _empty(device: torch.device) -> torch.Tensor:
    if device not in _EMPTY: _EMPTY[device] = torch.empty(0, device=device)
    return _EMPTY[device]


class _FusedMLS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, q, means, quats, sh1, rest, nbr, w, blend, beta, eps, group):
        ext = load_extension()
        e = _empty(means.device)
        cq = q.contiguous() if (blend and q is not None) else e
        om, oq, osh, R, U, V, sig, ps, tau = ext.forward(means, quats, sh1 if sh1 is not None else e, rest, nbr, w, t.contiguous(), cq, bool(blend), float(beta), int(group))
        ctx.save_for_backward(means, sh1 if sh1 is not None else e, oq, rest, nbr, w, cq, R, U, V, sig, ps, tau)
        ctx.meta = (bool(blend and q is not None), float(beta), float(eps), t.shape[0], int(group), sh1 is not None, q is not None)
        ctx.mark_non_differentiable(R, U, V, sig, ps, tau)
        return om, oq, (osh if sh1 is not None else None)

    @staticmethod
    def backward(ctx, g_means, g_quats, g_sh1):
        means, sh1, oq, rest, nbr, w, cq, R, U, V, sig, ps, tau = ctx.saved_tensors
        blend, beta, eps, M, group, has_sh, has_q = ctx.meta
        ext = load_extension(); e = _empty(means.device)
        prep = lambda g: g.contiguous().float() if g is not None else e
        grad_t, grad_q = ext.backward(prep(g_means), prep(g_quats), prep(g_sh1) if has_sh else e, means, sh1, oq, rest, nbr, w, cq,
                                      R, U, V, sig, ps, tau, blend, beta, eps, M, group)
        return grad_t, (grad_q if blend else None) if has_q else None, None, None, None, None, None, None, None, None, None, None


def fused_mls(means, quats, sh1, rest, nbr, w, t, q=None, blend=False, beta=0.0, eps=1e-6, group: int | None = None):
    """Kernel path. Inputs are made contiguous float32 / int32 here."""
    c = lambda x: None if x is None else x.detach().contiguous().float()
    nbr32 = nbr.contiguous() if nbr.dtype == torch.int32 else nbr.int().contiguous()
    om, oq, osh = _FusedMLS.apply(t, q, c(means), c(quats), c(sh1), c(rest), nbr32, c(w), bool(blend and q is not None and beta > 0), float(beta), float(eps), group or DEFAULT_GROUP)
    return om, oq, osh


def deform(means, quats, sh1, rest, nbr, w, t, q=None, blend=False, beta=0.0, eps=1e-6, group: int | None = None):
    use_kernel = means.is_cuda and t.dtype == torch.float32 and kernel_available()
    if use_kernel:
        return fused_mls(means, quats, sh1, rest, nbr, w, t, q, blend, beta, eps, group)
    return rigid_mls(means, quats, sh1, rest, nbr, w, t, q, blend, beta, eps)


class _FusedMLSRotation(torch.autograd.Function):
    """Rotation-only output of the fused kernel: R_i = polar(sum_k w_ik (p_k - p*)(p'_k - q*)^T)."""
    @staticmethod
    def forward(ctx, t, points, rest, nbr, w, eps):
        ext = load_extension(); e = _empty(points.device)
        quats = torch.zeros(points.shape[0], 4, device=points.device); quats[:, 0] = 1
        om, oq, _, R, U, V, sig, ps, tau = ext.forward(points, quats, e, rest, nbr, w, t.contiguous(), e, False, 0.0, 1)
        ctx.save_for_backward(points, oq, rest, nbr, w, R, U, V, sig, ps, tau)
        ctx.meta = (float(eps), t.shape[0])
        return R.view(-1, 3, 3)

    @staticmethod
    def backward(ctx, g_R):
        points, oq, rest, nbr, w, R, U, V, sig, ps, tau = ctx.saved_tensors
        eps, M = ctx.meta; e = _empty(points.device)
        grad_t, _ = load_extension().backward_r(e, e, e, g_R.reshape(-1, 9).contiguous().float(), points, e, oq, rest, nbr, w, e,
                                                R, U, V, sig, ps, tau, False, 0.0, eps, M, 1)
        return grad_t, None, None, None, None, None


def mls_rotation(rest, nbr, w, t, points=None, eps: float = 1e-6):
    """Per-point polar rotation of the weighted neighbourhood covariance (same definition and
    reflection fix as mls_ref.mls_rotation). `points` default to `rest` (e.g. controls rotating
    with their own graph neighbourhood). Fused kernel when available, else the reference."""
    points = rest if points is None else points
    if points.is_cuda and t.dtype == torch.float32 and kernel_available():
        nbr32 = nbr.contiguous() if nbr.dtype == torch.int32 else nbr.int().contiguous()
        c = lambda x: x.detach().contiguous().float()
        return _FusedMLSRotation.apply(t, c(points), c(rest), nbr32, c(w), float(eps))
    from ring_init.deform.mls_ref import mls_rotation as ref
    return ref(rest, nbr, w, t)[0]
