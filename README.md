# CourtFlow-GS

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

## Complete workflow: train, store, and play a model

This is the intended end-to-end workflow. The training machine needs the full stack (SAM2, MASt3R,
videos, and a CUDA GPU). The playback machine needs only Python, CUDA PyTorch, gsplat, ffmpeg, the
`ring_init` package, and the exported `playback/` directory.

### 1. Prepare a local training configuration

Copy the example configuration and set the paths to the SAM2 and MASt3R checkouts/checkpoints. Add
the playback-export flag before starting the run:

```bash
cp ring_init/configs/config.example.json ring_init/configs/basketball.local.json
# Edit basketball.local.json: set SAM2/MASt3R paths, then add:
#   "v2_export_playback": true

python -m ring_init.doctor --compile
```

### 2. Train and export

Point `--dataset-dir` to the raw capture containing `cameras/view_000.mp4` through
`view_035.mp4` and `calibration/`:

```bash
export PYTHONPATH=.:third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r
python -m ring_init.e2e --scene basketball --dataset-dir /datasets/basketball_capture \
  --config ring_init/configs/basketball.local.json --out-dir /outputs/courtflow \
  --frames 0:700
```

Stage A builds the canonical frame-0 model. Stage B tracks/refines the sequence and, because
`v2_export_playback` is enabled, writes the portable model package here:

```text
/outputs/courtflow/basketball/stage_b_v2/playback/
```

The package is the trained model for playback. It includes its own calibration in `meta.json`,
background Gaussian keyframes, and dynamic player/ball Gaussians for every frame; it does not
refer back to the training output, videos, config, or checkpoints.

### 3. Copy only the playback package

```bash
tar -C /outputs/courtflow/basketball/stage_b_v2 -czf courtflow_playback.tar.gz playback
scp courtflow_playback.tar.gz cuda-reviewer:/models/
```

On the reviewer machine, install a CUDA-compatible PyTorch build, gsplat, NumPy, ffmpeg, and this
repository/package. Then unpack the model anywhere:

```bash
mkdir -p /models/courtflow && tar -xzf /models/courtflow_playback.tar.gz -C /models/courtflow
```

### 4. Run inference from the stored model

```bash
python -m ring_init.play --package /models/courtflow/playback \
  --camera view:13 --frames 0:700 --out /models/courtflow/view13.mp4

# Measure package loading and GPU drawing separately from video encoding.
python -m ring_init.play --package /models/courtflow/playback \
  --camera view:13 --frames 0:700 --benchmark
```

No `--scene`, training configuration, or dataset path is supplied to `play`: `--package`
identifies the model. A physical capture camera is selected with `--camera view:<0-35>`.

## Track a sequence

```bash
export PYTHONPATH=.:third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r
python -m ring_init.run --scene basketball --stage b --frames 0:700 \
  --config ring_init/configs/basketball.local.json
```

Stage B writes online orbit and held-out-view videos, metrics, control-position history, resumable
state checkpoints, and baked PLY checkpoints under
`<out_root>/<scene>/stage_b_v2/`. Resume a stopped production run from a saved state with
`--resume-from stage_b_v2:FRAME`.

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

## Baked inspection

Rendering a saved checkpoint does not run tracking, SAM2, MLS, or optimization. It loads one
state (one instant of the reconstructed sequence) onto the GPU once and supports physical cameras
or a rig-following orbit. This is an inspection tool, not whole-sequence playback: for an animated
result, use the per-frame held-out and orbit videos published by the E2E run.

```bash
python -m ring_init.render_baked --scene basketball \
  --config ring_init/configs/basketball.local.json --tag stage_b_v2 \
  --checkpoint 699 --camera orbit --frames 0:700 --fps 25 --scale 0.5 \
  --out final_orbit.mp4
```

On the RTX 3080 test system, the final 2.0M-Gaussian checkpoint reaches 229 FPS for GPU-only
view-dependent colour evaluation plus rasterization at 937x527.  GPU-to-video encoding is a
separate integration concern; the included exporter uses NVENC when available.

## Forward-only playback

`ring_init.play` renders an exported playback package without loading the training pipeline. It
combines the player state for each requested frame with the latest background keyframe, evaluates
degree-1 colour coefficients, and rasterizes with gsplat:

```bash
python -m ring_init.play --package playback/ --camera view:13 \
  --frames 0:700 --out view13.mp4
python -m ring_init.play --package playback/ --camera view:13 --benchmark
```

The package contains `meta.json` (calibration, frame range, and FPS),
`background_XXXXXX.npz` files at keyframes, and `players_XXXXXX.npz` files for every frame.
Each array file stores only the parameters required to draw: means, rotations, log-scales, opacity
logits, degree-0/1 colour coefficients, and instance IDs.
Create it during the tracking run with
`--set v2_export_playback=true`; the resulting directory is
`<out_root>/<scene>/stage_b_v2/playback/` and can be copied unchanged to another CUDA machine.

## Repository layout

- `ring_init/stage_a.py` — canonical scene build.
- `ring_init/stage_b.py`, `ring_init/deform/scene_track.py` — selected online method.
- `ring_init/deform/csrc/mls_kernel.cu` — fused rigid MLS and local rotations.
- `ring_init/gs/csrc/fused_ssim/` — vendored fused SSIM kernel and license.
- `ring_init/render_baked.py` — offline inspection of one saved checkpoint.
- `ring_init/play.py` — forward-only playback of an exported dynamic package.
- `ring_init/tests/` — calibration, control, raster, MLS, and speed-regression tests.

The legacy v1 person-only tracker is retained only for reproducibility; use `stage_b_mode="scene"`
(the default) for this method.

## Final report

The authoritative results, protocol, runtime measurements, and limitations are in
[`report/courtflow_gs_report.pdf`](report/courtflow_gs_report.pdf). They describe the final
700-frame, 12-view run on one RTX PRO 6000: 20.43 dB mean held-out PSNR at 1080p and about
119 minutes end to end. `REPORT.md` is retained only as an archived early draft and must not be
used for final numbers.
