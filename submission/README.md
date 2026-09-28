# CourtFlow-GS submission

## Contents

- `courtflow_gs_report.pdf` — final report.
- `videos/` — final held-out-camera and orbit renders, plus the uncapped comparison.
- `metrics/` — final-run summary JSON.
- `code/` — source at commit `1e03d36` (or later if this package was rebuilt).

## Reproducing playback

The included videos are immediately viewable. A complete portable playback model requires one
final tracker rerun with `"v2_export_playback": true` in the training config. That run writes
`stage_b_v2/playback/`; copy that directory beside this README and run:

```bash
python -m ring_init.play --package playback --camera view:13 --frames 0:700 --out view13.mp4
```

The playback host needs a CUDA-capable NVIDIA GPU, CUDA PyTorch, gsplat, ffmpeg, NumPy, and the
`code/` source tree. No dataset, SAM2, MASt3R, detector, optimizer, or training checkpoints are
required at inference time.
