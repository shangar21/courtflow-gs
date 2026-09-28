#!/bin/bash
# Like cfenv.sh, but imports ring_init from /home/ubuntu/courtflow-gs-next (cwd is first on sys.path for python -m).
export PATH=/home/ubuntu/courtflow-env/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/courtflow-env/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/ubuntu/courtflow-gs-next:/home/ubuntu/third_party/MAtCha/mast3r:/home/ubuntu/third_party/MAtCha/mast3r/dust3r
export PYTHONUNBUFFERED=1 CUDA_HOME=/home/ubuntu/courtflow-env TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8
export CPATH=/home/ubuntu/courtflow-env/targets/x86_64-linux/include${CPATH:+:$CPATH}
cd /home/ubuntu/courtflow-gs-next
exec "$@"
