#!/bin/bash
# 1080p: Gaussian cap 2M from frame 0 (checkpoints every 50), plus a 3M branch from its frame-250 checkpoint.
exec 9>/home/ubuntu/logs/cap_sweep.lock; flock -n 9 || { echo "already running"; exit 0; }
L=/home/ubuntu/logs; B=/home/ubuntu/outputs/courtflow/basketball; R=/home/ubuntu/courtflow-gs
st() { echo "[$(date -u +%FT%TZ)] $*" >> $L/cap_sweep_status.txt; }
cd $R && git fetch /home/ubuntu/pro6000.bundle pro6000-run >> $L/git_ff.log 2>&1 && git merge --ff-only FETCH_HEAD >> $L/git_ff.log 2>&1 || { st "FF FAILED"; exit 1; }
[ "$(git rev-parse --short HEAD)" = "8492375" ] || { st "unexpected HEAD $(git rev-parse --short HEAD)"; exit 1; }
st "HEAD $(git log --oneline -1)"
CFG=$B/e2e_config.json
run() { /home/ubuntu/bin/cfenv.sh python -m ring_init.run --scene basketball --stage b --frames 0:700 --config $CFG --set v2_growth_cap=0 "$@"; }
rm -rf $B/stage_b_cap2m $B/stage_b_cap3m
st "cap 2M start"; ( /usr/bin/time -v bash -c "$(declare -f run); CFG=$CFG; run --tag stage_b_cap2m --set v2_max_gaussians=2000000" > $L/cap2m.log 2>&1; st "cap 2M exit $?" ) &
until [ -f $B/stage_b_cap2m/state/frame_000250.pt ] || ! pgrep -f "tag stage_b_cap2m" > /dev/null; do sleep 30; done
sleep 20   # let the checkpoint write finish
if [ -f $B/stage_b_cap2m/state/frame_000250.pt ]; then
  st "cap 3M branch start from stage_b_cap2m:250"
  ( /usr/bin/time -v bash -c "$(declare -f run); CFG=$CFG; run --tag stage_b_cap3m --resume-from stage_b_cap2m:250 --set v2_max_gaussians=3000000" > $L/cap3m.log 2>&1; st "cap 3M exit $?" ) &
else st "cap 2M ended before frame 250; no 3M branch"; fi
wait; st "ALL DONE"
