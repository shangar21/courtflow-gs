#!/bin/bash
# Usage: cfenv.sh <command...>  -- runs a command in the CourtFlow remote environment
export PATH=/home/ubuntu/courtflow-env/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/courtflow-env/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/ubuntu/courtflow-gs:/home/ubuntu/third_party/MAtCha/mast3r:/home/ubuntu/third_party/MAtCha/mast3r/dust3r
export PYTHONUNBUFFERED=1
export CUDA_HOME=/home/ubuntu/courtflow-env
export CPATH=/home/ubuntu/courtflow-env/targets/x86_64-linux/include${CPATH:+:$CPATH}
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
cd /home/ubuntu/courtflow-gs
exec "$@"
