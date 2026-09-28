set -u
rm -rf /home/ubuntu/gitdry && cp -a /home/ubuntu/courtflow-gs /home/ubuntu/gitdry && cd /home/ubuntu/gitdry
git diff report/courtflow_gs_report.tex | grep '^+[^+]' | cut -c2- > /tmp/remote_added_lines.txt
# --- exact orchestrator sequence ---
git stash push -m "pre-pro6000-run hotfixes" || echo "STASH FAILED"
rm -f ring_init/tests/test_frames.py
git fetch /home/ubuntu/pro6000.bundle pro6000-run:pro6000-run || echo "FETCH FAILED"
git checkout pro6000-run || echo "CHECKOUT FAILED"
# --- verification ---
echo "HEAD: $(git log --oneline -1)"; echo "status:"; git status --short | head
echo "stash: $(git stash list)"
missing=0; while IFS= read -r l; do grep -qxF -- "$l" report/courtflow_gs_report.tex || { echo "MISSING remote report line: $l"; missing=1; }; done < /tmp/remote_added_lines.txt; echo "remote report edits preserved in branch: $([ $missing = 0 ] && echo yes || echo NO)"
for f in doctor.py io/frames.py; do git diff stash@{0} -- ring_init/$f | grep -c '^-[^-]' | xargs echo "lines only in server's $f (not in branch):"; done
ls ring_init/deform/csrc/build/*.so ring_init/configs/config.local.json 2>&1 | head -3
