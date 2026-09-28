#!/bin/bash
exec 9>/home/ubuntu/logs/resume_test.lock; flock -n 9 || exit 0
cd /home/ubuntu/courtflow-gs-next && rm -rf ring_init && tar xzf /home/ubuntu/ring_init_next.tgz
B=/home/ubuntu/outputs/courtflow/basketball; CFG=$B/e2e_config.json; rm -rf $B/rt_full $B/rt_res
run() { /home/ubuntu/bin/cfenv_next.sh python -m ring_init.run --scene basketball --stage b --frames 0:60 --config $CFG \
          --set v2_save_state_every=20 --set v2_growth_cap=0 --set v2_max_gaussians=800000 "$@"; }
run --tag rt_full > /home/ubuntu/logs/rt_full.log 2>&1; echo "rt_full exit $?" >> /home/ubuntu/logs/resume_test_status.txt
run --tag rt_res --resume-from rt_full:20 > /home/ubuntu/logs/rt_res.log 2>&1; echo "rt_res exit $?" >> /home/ubuntu/logs/resume_test_status.txt
echo DONE >> /home/ubuntu/logs/resume_test_status.txt
