#!/bin/bash
# Full 700-frame 4K run, started only after the fixed 1080p run has finished successfully.
exec 9>/home/ubuntu/logs/run4k_full.lock; flock -n 9 || { echo "already running"; exit 0; }
L=/home/ubuntu/logs; D=/home/ubuntu/outputs/courtflow_4k/basketball; R=/home/ubuntu/courtflow-gs
st() { echo "[$(date -u +%FT%TZ)] $*" >> $L/run4k_status.txt; }
st "waiting for fixed 1080p run"
until [ -f $L/fixed1080_end.txt ]; do sleep 60; done
grep -q "Exit status: 0" $L/fixed1080.log || { st "fixed 1080p run failed; 4K not started"; exit 1; }
cd $R && [ "$(git rev-parse --short HEAD)" = "355d338" ] || { st "unexpected HEAD $(git rev-parse --short HEAD)"; exit 1; }
[ -f $R/ring_init/configs/config.4k.json ] || { st "config.4k.json missing"; exit 1; }
# Range-dependent Stage B caches from the 30-frame smoke must not leak into the 700-frame run.
[ -d $D/stage_b_v2_smoke30 ] || mv $D/stage_b_v2 $D/stage_b_v2_smoke30
rm -rf $D/ball $D/canonical_ball $D/stage_b_v2
st "starting 4K 0:700 at HEAD $(git log --oneline -1)"
nohup nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $L/4k_full_vram.csv 2>&1 < /dev/null &
VPID=$!
date -u +%FT%TZ > $L/4k_full_start.txt
/usr/bin/time -v /home/ubuntu/bin/cfenv.sh python -m ring_init.e2e --scene basketball --dataset-dir /home/ubuntu/datasets/basketball \
  --config $R/ring_init/configs/config.4k.json --out-dir /home/ubuntu/outputs/courtflow_4k --frames 0:700 > $L/4k_full.log 2>&1
date -u +%FT%TZ > $L/4k_full_end.txt
kill $VPID
grep -q "Exit status: 0" $L/4k_full.log && st "4K DONE OK" || st "4K FAILED (see 4k_full.log)"
