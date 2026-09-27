"""Fused SSIM (vendored rahul-goel/fused-ssim CUDA kernel, MIT; see csrc/fused_ssim/).

Same definition as `ring_init.gs.train.ssim_map`: 11x11 Gaussian window (sigma 1.5), zero padding,
C1 = 0.01^2, C2 = 0.03^2, per-pixel map of the input shape. Gradients flow to the first image only
(the second is the target). JIT-built on first use; set RING_FUSED_SSIM_DISABLE=1 to force the
PyTorch path."""
from __future__ import annotations
import os
from pathlib import Path
import torch

_CSRC = Path(__file__).parent / "csrc" / "fused_ssim"
_EXT = None
_EXT_ERROR: Exception | None = None
C1, C2 = 0.01 ** 2, 0.03 ** 2


def load_extension(verbose: bool = False):
    global _EXT, _EXT_ERROR
    if _EXT is not None: return _EXT
    if _EXT_ERROR is not None: raise _EXT_ERROR
    try:
        import torch.utils.cpp_extension as ce
        from ring_init.deform.mls import _matching_cuda_home
        home = _matching_cuda_home()
        if home is None: raise RuntimeError(f"No CUDA toolkit matching torch CUDA {torch.version.cuda} found for the JIT build")
        ce.CUDA_HOME = home; os.environ["CUDA_HOME"] = home
        if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability(); os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        build = _CSRC / "build"; build.mkdir(parents=True, exist_ok=True)
        _EXT = ce.load(name="ring_fused_ssim", sources=[str(_CSRC / "ext.cpp"), str(_CSRC / "ssim.cu")], build_directory=str(build),
                       extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=verbose)
        return _EXT
    except Exception as error:
        _EXT_ERROR = error
        raise


def available() -> bool:
    if os.environ.get("RING_FUSED_SSIM_DISABLE") == "1" or not torch.cuda.is_available(): return False
    try:
        load_extension(); return True
    except Exception:
        return False


class _FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2):
        ext = load_extension()
        ssim_map, d_mu1, d_s1, d_s12 = ext.fusedssim(C1, C2, img1, img2, img1.requires_grad)
        ctx.save_for_backward(img1, img2, d_mu1, d_s1, d_s12)
        return ssim_map

    @staticmethod
    def backward(ctx, grad_map):
        img1, img2, d_mu1, d_s1, d_s12 = ctx.saved_tensors
        grad = load_extension().fusedssim_backward(C1, C2, img1, img2, grad_map.contiguous(), d_mu1, d_s1, d_s12)
        return grad, None


def fused_ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[3,H,W] or [B,3,H,W] float32 CUDA images -> SSIM map of the same shape."""
    single = x.dim() == 3
    if single: x, y = x[None], y[None]
    m = _FusedSSIMMap.apply(x.float().contiguous(), y.detach().float().contiguous())
    return m[0] if single else m
