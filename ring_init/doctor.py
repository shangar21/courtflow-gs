"""Validate a deployment GPU and optionally precompile CourtFlow-GS CUDA extensions.

Run this once per machine/image after installing a PyTorch build that supports the local CUDA
driver and GPU architecture:

    python -m ring_init.doctor --compile

The fused MLS and SSIM extensions select ``TORCH_CUDA_ARCH_LIST`` from the installed GPU when it
is unset, so the same source distribution can build on an RTX 3080, RTX PRO 6000, or another
CUDA-capable NVIDIA device.  A compatible nvcc toolkit matching ``torch.version.cuda`` is still
required for compilation.
"""
from __future__ import annotations
import argparse
import os
import sys
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compile", action="store_true", help="JIT-build fused MLS and fused SSIM for this GPU")
    args = ap.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build for this machine.")
    index = torch.cuda.current_device(); props = torch.cuda.get_device_properties(index); cc = torch.cuda.get_device_capability(index)
    print(f"GPU {index}: {props.name}; compute capability {cc[0]}.{cc[1]}; VRAM {props.total_memory / 2**30:.1f} GiB")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
        print(f"TORCH_CUDA_ARCH_LIST set to {os.environ['TORCH_CUDA_ARCH_LIST']} for this process")
    else:
        print(f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    if args.compile:
        from ring_init.deform import mls
        from ring_init.gs import fused_ssim
        mls.load_extension(verbose=True); fused_ssim.load_extension(verbose=True)
        print("Fused MLS and SSIM extensions compiled successfully.")
    else:
        print("GPU preflight passed. Run with --compile to prewarm custom CUDA extensions.")


if __name__ == "__main__":
    main()
