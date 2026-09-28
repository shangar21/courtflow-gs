#!/bin/bash
# Unattended sequence: wait for the 1080p run -> raster benchmarks -> checkout pro6000-run
# -> 4K 30-frame smoke with automatic checks -> full 700-frame 4K run.
set -u
L=/home/ubuntu/logs; R=/home/ubuntu/courtflow-gs; ENV=/home/ubuntu/bin/cfenv.sh
OUT4K=/home/ubuntu/outputs/courtflow_4k; CFG4K=$R/ring_init/configs/config.4k.json
step() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $L/orchestrate_status.txt; }
fail() { step "FAILED: $*"; exit 1; }

step "waiting for 1080p run"
until grep -q "Exit status" $L/full.log; do sleep 60; done
date -u +%FT%TZ > $L/full_end.txt
pkill -f "nvidia-smi --query-gpu" || true
grep -q "Exit status: 0" $L/full.log || fail "1080p run exited non-zero (see full.log)"
step "1080p run finished OK"

step "raster benchmarks (idle GPU)"
$ENV python /home/ubuntu/bench_render.py --config /home/ubuntu/outputs/courtflow/basketball/e2e_config.json \
    --widths 937,1920,3840 --json-out $L/bench_pro_final.json > $L/bench.log 2>&1 || step "WARN: PRO checkpoint bench failed"
$ENV python /home/ubuntu/bench_render.py --config /home/ubuntu/outputs/courtflow/basketball/e2e_config.json \
    --ply /home/ubuntu/bench_3080_frame299.ply --widths 937,1920,3840 --json-out $L/bench_3080_model.json >> $L/bench.log 2>&1 || step "WARN: 3080 model bench failed"

step "checking out pro6000-run"
cd $R || fail "no repo"
git stash push -m "pre-pro6000-run hotfixes" >> $L/git.log 2>&1
rm -f ring_init/tests/test_frames.py   # untracked copy of a file the branch adds (identical content)
git fetch /home/ubuntu/pro6000.bundle pro6000-run:pro6000-run >> $L/git.log 2>&1 || fail "git fetch"
git checkout pro6000-run >> $L/git.log 2>&1 || fail "git checkout"
[ "$(git rev-parse HEAD)" = "$(git rev-parse pro6000-run)" ] || fail "HEAD mismatch"
step "HEAD $(git log --oneline -1)"
cp /home/ubuntu/config.4k.json $CFG4K
$ENV python -m ring_init.doctor --compile > $L/doctor_4k.log 2>&1 || fail "doctor --compile"

step "4K smoke (Stage A + 30 frames)"
nohup nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $L/4k_vram.csv 2>&1 < /dev/null &
date -u +%FT%TZ > $L/4k_start.txt
/usr/bin/time -v $ENV python -m ring_init.e2e --scene basketball --dataset-dir /home/ubuntu/datasets/basketball \
    --config $CFG4K --out-dir $OUT4K --frames 0:30 > $L/4k_smoke.log 2>&1
grep -q "Exit status: 0" $L/4k_smoke.log || fail "4K smoke exited non-zero (see 4k_smoke.log)"
$ENV python - <<'PY' || fail "4K smoke checks"
import json, subprocess, sys
d = "/home/ubuntu/outputs/courtflow_4k/basketball/stage_b_v2"
s = json.load(open(f"{d}/summary.json")); ok = True
for k, lo in (("eval_heldout_psnr", 17.0), ("eval_heldout_person_psnr", 18.0), ("eval_train_iou", 0.4)):
    print(k, s[k]); ok &= s[k] > lo
for v in ("videos/heldout_view13.mp4", "videos/orbit360.mp4", "final_renders/animated_eval_view13.mp4"):
    wh = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", f"{d}/{v}"], capture_output=True, text=True).stdout.strip()
    print(v, wh); ok &= wh.startswith("3840,2160")
sys.exit(0 if ok else 1)
PY
step "4K smoke passed; clearing range-dependent Stage B caches"
cp -r $OUT4K/basketball/stage_b_v2 $OUT4K/basketball/stage_b_v2_smoke30
rm -rf $OUT4K/basketball/ball $OUT4K/basketball/canonical_ball $OUT4K/basketball/stage_b_v2

step "4K full run (700 frames)"
date -u +%FT%TZ > $L/4k_full_start.txt
/usr/bin/time -v $ENV python -m ring_init.e2e --scene basketball --dataset-dir /home/ubuntu/datasets/basketball \
    --config $CFG4K --out-dir $OUT4K --frames 0:700 > $L/4k_full.log 2>&1
date -u +%FT%TZ > $L/4k_full_end.txt
pkill -f "nvidia-smi --query-gpu" || true
grep -q "Exit status: 0" $L/4k_full.log || fail "4K full run exited non-zero (see 4k_full.log)"
$ENV python /home/ubuntu/bench_render.py --config $OUT4K/basketball/e2e_config.json --widths 937,1920,3840 \
    --json-out $L/bench_pro_4k_final.json >> $L/bench.log 2>&1 || step "WARN: 4K checkpoint bench failed"
step "ALL DONE"
