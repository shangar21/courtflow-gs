#!/bin/bash
# Stage B keyframe variants on the cached 1080p data, frames 0:300, run concurrently from courtflow-gs-next.
exec 9>/home/ubuntu/logs/sweep.lock; flock -n 9 || { echo 'sweep already running'; exit 0; }
rm -rf /home/ubuntu/outputs/courtflow/basketball/sweep_A /home/ubuntu/outputs/courtflow/basketball/sweep_B /home/ubuntu/outputs/courtflow/basketball/sweep_C
cd /home/ubuntu/courtflow-gs-next && rm -rf ring_init && tar xzf /home/ubuntu/ring_init_next.tgz
CFG=/home/ubuntu/outputs/courtflow/basketball/e2e_config.json
run() { tag=$1; shift
  { /home/ubuntu/bin/cfenv_next.sh python -c "import ring_init; print('ring_init from', ring_init.__file__)"
    /home/ubuntu/bin/cfenv_next.sh python -m ring_init.run --scene basketball --stage b --frames 0:300 --config $CFG --tag $tag "$@"; } > /home/ubuntu/logs/$tag.log 2>&1
  echo "$tag exit $?" >> /home/ubuntu/logs/sweep_status.txt; }
run sweep_A --set v2_keyframe_dynamic_only=true --set v2_growth_cap=0.5 &
run sweep_B --set v2_growth_cap=0.5 &
run sweep_C --set v2_keyframe_dynamic_only=true --set v2_growth_cap=0.5 --set v2_refine_dynamic_only=true &
wait; echo "ALL DONE" >> /home/ubuntu/logs/sweep_status.txt
