#!/bin/bash
# 1080p 3M branch (from the finished 2M run) and the final 4K run with a 3M cap, in parallel.
exec 9>/home/ubuntu/logs/final_runs.lock; flock -n 9 || { echo "already running"; exit 0; }
L=/home/ubuntu/logs; R=/home/ubuntu/courtflow-gs
st() { echo "[$(date -u +%FT%TZ)] $*" >> $L/final_runs_status.txt; }
cd $R && git fetch /home/ubuntu/pro6000.bundle pro6000-run >> $L/git_ff.log 2>&1 && git merge --ff-only FETCH_HEAD >> $L/git_ff.log 2>&1 || { st "FF FAILED"; exit 1; }
[ "$(git rev-parse --short HEAD)" = "65dabe3" ] || { st "unexpected HEAD $(git rev-parse --short HEAD)"; exit 1; }
st "HEAD $(git log --oneline -1)"
B1=/home/ubuntu/outputs/courtflow/basketball; B4=/home/ubuntu/outputs/courtflow_4k/basketball
rm -rf $B1/stage_b_cap3m $B4/stage_b_cap3m
( /usr/bin/time -v /home/ubuntu/bin/cfenv.sh python -m ring_init.run --scene basketball --stage b --frames 0:700 --config $B1/e2e_config.json \
    --tag stage_b_cap3m --resume-from stage_b_cap2m:250 --set v2_growth_cap=0 --set v2_max_gaussians=3000000 > $L/cap3m.log 2>&1; st "1080p 3M exit $?" ) &
( nohup nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $L/4k_cap3m_vram.csv 2>&1 < /dev/null & VPID=$!
  /usr/bin/time -v /home/ubuntu/bin/cfenv.sh python -m ring_init.run --scene basketball --stage b --frames 0:700 --config $B4/e2e_config.json \
    --tag stage_b_cap3m --set v2_growth_cap=0 --set v2_max_gaussians=3000000 > $L/4k_cap3m.log 2>&1; st "4K 3M exit $?"; kill $VPID ) &
st "both started"; wait; st "ALL DONE"
