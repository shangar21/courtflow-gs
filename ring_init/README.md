# Ring dynamic Gaussian capture — Stage A (frame 0)

Known-pose 12-camera reconstruction of every on-court person (plus a frozen static background)
from a single synchronized frame. Cameras are never optimized; no SfM is run.
Design: `docs/superpowers/specs/2026-09-26-ring-init-stage-a-design.md`; spec deviations:
`DEVIATIONS.md`.

```bash
export PYTHONPATH=.:third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r
python -m ring_init.run --scene basketball --stage a --config ring_init/configs/basketball.json
python -m ring_init.run ... --force hulls,fuse      # recompute steps (and everything cached after them manually)
python -m ring_init.run ... --until instances       # stop early
python -m pytest -q ring_init/tests
```

## Steps (outputs under `out/<scene>/<step>/`, each cached and resumable)

| step | what | debug output |
|---|---|---|
| calib | undistort, COLMAP-format export, ring/frusta figure | `calib/calibration_frusta.png` |
| match_full | MASt3R 512-px on 24 ring pairs (adjacent + skip-one), known-P DLT, cheirality / 1.5 px / 3 deg filters | `match_full/pair_*.npz` |
| floor | RANSAC floor plane, court extent | `floor/floor.json` |
| persons | full-res person detector boxes -> SAM2 masks (court only); scene scale from standing heights | `persons/cam_*.npz` |
| instances | occupancy carving of union person masks over the court -> 3D components = instances -> per-view SAM2 masks with depth-ordered exclusive labels | `instances/masks/cam_XX/instance_YYY.png`, `contact_sheet.jpg`, `instances.json` |
| hulls | occlusion-aware per-instance visual hull (1 cm) | `hulls/instance_*.npz` |
| match_crops | per-instance crop MASt3R, DLT, hull carving | `match_crops/*.npz` |
| fuse | triangulated + hull-fill points, SOR, voxel grid, normals, median colours; background cloud | `fuse/instance_*.ply`, `fuse/background.ply` |
| train | per-instance gsplat t0 (crop rendering, masked L1 + D-SSIM, alpha BCE, sparse depth, scale ratio, DropGaussian, capped densification, mask pruning), then background | `train/instance_*/train_views.jpg`, `train_log.jsonl`, `stats.json` |
| export | canonical per-instance 3DGS `.ply` + `instance_ids.npy`, `manifest.json` | `canonical/` |
| eval | 24 held-out cameras (only place they are read): PSNR/SSIM/LPIPS full + person region, person IoU, floater mass, counts, times, FPS | `eval/metrics.json`, `eval/report.md`, `eval/view_*.jpg` |

All thresholds live in `config.py` (metric lengths in meters, converted by `units_per_meter`).

## Stage B baked-state inference

Stage B tracking is an offline optimisation process; it is not the playback renderer.  Each
saved `stage_b_*` checkpoint contains a complete, baked Gaussian state that can be loaded once
onto the GPU and rendered from an arbitrary calibrated camera without SAM2, control-point
optimisation, or MLS deformation.

```bash
PYTHONPATH=.:third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r \
python -m ring_init.render_baked --scene basketball --config ring_init/configs/basketball.json \
  --tag stage_b_diag_crop_ball --checkpoint 299 --camera orbit --frames 0:300 \
  --fps 25 --scale 0.5 --out final_orbit.mp4
```

Use `--camera view:13`, `--camera train:0`, or `--camera orbit`.  Omit `--checkpoint` to use
the latest saved state.  `render_baked` defaults to RTX NVENC output; use `--codec libx264` if
hardware encoding is unavailable.  The final diagnostic checkpoint is stored at
`stage_b_diag_crop_ball/ply/frame_000299/point_cloud.ply`; control positions and run metadata
remain alongside it for resuming optimisation.
