#!/bin/bash
# Final 1080p run with the growth cap; Stage A, frames, masks and ball are reused from the first run.
exec 9>/home/ubuntu/logs/run1080_fixed.lock; flock -n 9 || { echo "already running"; exit 0; }
L=/home/ubuntu/logs; B=/home/ubuntu/outputs/courtflow/basketball
cd /home/ubuntu/courtflow-gs || exit 1
git fetch /home/ubuntu/pro6000.bundle pro6000-run > $L/git_ff.log 2>&1 && git merge --ff-only FETCH_HEAD >> $L/git_ff.log 2>&1 || { echo "FF FAILED" >> $L/git_ff.log; exit 1; }
echo "HEAD $(git log --oneline -1)" >> $L/git_ff.log
grep -q "v2_growth_cap: float = 0.5" ring_init/config.py || { echo "cap default missing" >> $L/git_ff.log; exit 1; }
[ -d $B/stage_b_v2_uncapped ] || mv $B/stage_b_v2 $B/stage_b_v2_uncapped
nohup nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $L/fixed1080_vram.csv 2>&1 < /dev/null &
VPID=$!
date -u +%FT%TZ > $L/fixed1080_start.txt
/usr/bin/time -v /home/ubuntu/bin/cfenv.sh python -m ring_init.e2e --scene basketball --dataset-dir /home/ubuntu/datasets/basketball \
  --config /home/ubuntu/courtflow-gs/ring_init/configs/config.local.json --out-dir /home/ubuntu/outputs/courtflow --frames 0:700 > $L/fixed1080.log 2>&1
date -u +%FT%TZ > $L/fixed1080_end.txt
kill $VPID
