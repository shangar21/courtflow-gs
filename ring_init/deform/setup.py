"""Install the fused MLS kernel as `ring_mls_cuda` (the JIT path in mls.py is used otherwise).

    CUDA_HOME=/usr/local/cuda-<torch CUDA version> TORCH_CUDA_ARCH_LIST=8.6 \
        python ring_init/deform/setup.py install
"""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = Path(__file__).resolve().parent
setup(name="ring_mls_cuda",
      ext_modules=[CUDAExtension("ring_mls_cuda", [str(here / "csrc" / "mls_kernel.cu")],
                                 extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]})],
      cmdclass={"build_ext": BuildExtension})
