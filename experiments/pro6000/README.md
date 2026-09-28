# RTX PRO 6000 run scripts

The scripts used on the Brev RTX PRO 6000 instance to produce the report's measurements, kept as
they ran. Paths assume the instance layout (`/home/ubuntu/courtflow-gs`, outputs under
`/home/ubuntu/outputs/`); adjust them for another machine.

## Environment

| Script | Purpose |
|---|---|
| `cfenv.sh` | Run a command in the project environment (PATH, conda `libstdc++`, CUDA headers for gsplat's JIT build, `PYTHONUNBUFFERED`). |
| `cfenv_next.sh` | Same, but imports `ring_init` from a second checkout (`courtflow-gs-next`) so experiments never touch a running job's code. |

## Runs

| Script | What it ran |
|---|---|
| `orchestrate.sh` | Unattended chain: wait for the 1080p run, benchmark, check out the branch, 4K smoke test with automatic checks, full 4K run. |
| `sweep.sh` | Three keyframe variants on frames 0–299 (background frozen, growth capped, both); chose the growth cap. |
| `run1080_fixed.sh` | Final 1080p 700-frame run with the growth cap, reusing cached Stage A, frames, masks and ball. |
| `run4k_full.sh` | Full 700-frame native-4K run, started only after the 1080p run succeeded. |
| `cap_sweep.sh` | 2M-cap run from frame 0 with checkpoints, plus a 3M branch from its frame-250 checkpoint. |
| `final_runs.sh` | 1080p 3M branch and the 4K 3M run in parallel. |
| `resume_test.sh`, `check_resume.py` | Real-data test that a run resumed from a checkpoint matches an uninterrupted one. |
| `git_dryrun.sh` | Dry run of the server checkout on a copy of the repo (conflict check). |

## Measurements behind the report

| Script | Report claim |
|---|---|
| `bench_render.py` | Render (draw) time per frame at 937×527, 1080p and 4K, CUDA-synchronized on an idle GPU. |
| `fair_compare.py` | 1080p vs 4K models scored on identical 1080p test images. |
| `heldout_grid.py` | Best / median / worst test-camera grids, chosen by score rather than by hand. |
| `per_player.py` | Per-person PSNR on identical pixels, plus the tracker-style camera-averaged score. |
| `bake_test.py`, `bake_check.py` | Bake time and controls-only playback quality; player positions reproduced by baking. |
| `cmp_caps.py`, `cmp_sweep.py` | Whole-video comparison of Gaussian budgets and of the keyframe variants. |
| `check_4k_frames.py`, `check_par_masks.py`, `prof_preproc.py`, `prof_nvdec.sh`, `gsplat_check.py` | Checks for the fast 4K data path, parallel SAM2 equivalence, preprocessing profile, NVDEC timing, gsplat build. |
