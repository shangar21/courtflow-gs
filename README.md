# Ring Dynamic Gaussians

A focused multi-view dynamic Gaussian pipeline for a calibrated camera ring.  The selected
method reconstructs a canonical scene, tracks dynamic instances using control points and fused
rigid MLS, performs focused inter-frame Gaussian refinement, and exports baked checkpoints for
fast novel-view rendering.

## Final method

For each frame, the tracker:

1. re-prompts SAM2 masks every fifth frame from the current multi-view render;
2. optimizes per-instance global motion plus local control-point offsets;
3. deforms dynamic Gaussians using fused rigid MLS over eight same-group control points;
4. composes them over a cached static background;
5. runs 100 crop-focused Gaussian refinement iterations between full 500-iteration keyframes;
6. bakes the live state and emits checkpoint/video diagnostics.

The defaults in `ring_init/config.py` represent this selected method.  The frame-0 canonical
scene is a required input to Stage B; it is produced by Stage A or supplied as an exported
`canonical/` (and optionally `canonical_ball/`) package.

## Install

Use Python 3.10+ and a CUDA-enabled PyTorch build.  Then install the Python dependencies:

```bash
pip install -r ring_init/requirements.txt
pip install -e .
```

MASt3R and SAM2 are external dependencies.  Point the scene config at their checkouts and
checkpoints; see `ring_init/configs/basketball.json` for the required fields.  Do not commit
camera videos, calibration assets, trained PLYs, or local path-bearing config files.

### Multi-machine / RTX PRO deployment

The source package contains the fused MLS and SSIM CUDA sources, rather than a 3080-specific
binary.  On every target machine—including an RTX PRO 6000—install a PyTorch/CUDA combination
that supports that GPU, then preflight and compile for its detected compute capability:

```bash
python -m ring_init.doctor --compile
```

This compiles the extensions locally and caches them for the target architecture.  Build a source
distribution with `python -m build`; `pyproject.toml` and `MANIFEST.in` include the CUDA sources.

## Track a sequence

```bash
export PYTHONPATH=.:third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r
python -m ring_init.run --scene basketball --stage b --frames 0:300 \
  --config ring_init/configs/basketball.local.json
```

Stage B writes an online orbit plus held-out-view videos, metrics, control-position history, and
baked PLY checkpoints under `<out_root>/<scene>/stage_b_v2/`.

## One-command E2E run

`ring_init.e2e` is the release entry point. It takes the training-data directory, runs canonical
Stage A plus the selected Stage B method, evaluates held-out views, and publishes an animated
held-out-view video and orbit rendered from the state at each video frame. A single baked
checkpoint is one instant in time; it is not repeated as a misleading static ``final video.''

For a raw capture root containing `cameras/view_000.mp4` … `view_035.mp4` and
`calibration/cameras.txt` + `images.txt`:

```bash
python -m ring_init.e2e --scene basketball --dataset-dir /datasets/basketball_capture \
  --config ring_init/configs/config.example.json --out-dir /outputs/ring_final --frames 0:700
```

The bootstrap adapter derives the 12 training cameras, all-view pinhole calibration, identity
photometric metadata, and frame-0 evaluation images from that raw layout.  The local config only
needs SAM2/MASt3R paths for this case.  Use `--prepare-only` to validate this setup without
training. Copy `ring_init/configs/config.example.json` to a `.local.json` file and set your SAM2
paths before a real run. The command writes its fully resolved configuration to
`<out-dir>/<scene>/e2e_config.json` for reproducibility.

## Baked inference

Rendering a saved checkpoint does not run tracking, SAM2, MLS, or optimization. It loads one
state (one instant of the reconstructed sequence) onto the GPU once and supports physical cameras
or a rig-following orbit. For an animated result, use the per-frame held-out and orbit videos
published by the E2E run:

```bash
python -m ring_init.render_baked --scene basketball \
  --config ring_init/configs/basketball.local.json --tag stage_b_diag_crop_ball \
  --checkpoint 299 --camera orbit --frames 0:300 --fps 25 --scale 0.5 \
  --out final_orbit.mp4
```

On the RTX 3080 test system, the final 2.0M-Gaussian checkpoint reaches 229 FPS for GPU-only
view-dependent colour evaluation plus rasterization at 937x527.  GPU-to-video encoding is a
separate integration concern; the included exporter uses NVENC when available.

## Repository layout

- `ring_init/stage_a.py` — canonical scene build.
- `ring_init/stage_b.py`, `ring_init/deform/scene_track.py` — selected online method.
- `ring_init/deform/csrc/mls_kernel.cu` — fused rigid MLS and local rotations.
- `ring_init/gs/csrc/fused_ssim/` — vendored fused SSIM kernel and license.
- `ring_init/render_baked.py` — offline checkpoint inference.
- `ring_init/tests/` — calibration, control, raster, MLS, and speed-regression tests.

The legacy v1 person-only tracker is retained only for reproducibility; use `stage_b_mode="scene"`
(the default) for this method.
