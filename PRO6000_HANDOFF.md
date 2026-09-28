# CourtFlow-GS RTX PRO 6000 handoff

Last updated: 2026-09-28. Status: **all runs complete; report complete.**

## Final method and results

- 1080p working resolution, keyframe growth capped at 2 million Gaussians (`v2_max_gaussians`
  default). Test PSNR 20.43 dB over all 700 frames on the 24 held-out cameras; about 119 minutes
  end to end on one RTX PRO 6000; a new view renders in about 3 ms at 1080p and 3.8 ms at 4K.
- Native 4K: about 151 minutes end to end and worse on identical 1080p test images
  (19.74 dB with 1.2M Gaussians, 19.30 dB with 3M).
- Uncapped growth collapsed late in the video (6.86M Gaussians, 16.1 dB at frame 699); a 3M cap
  gives sharper players in typical views but lower whole-scene and per-person scores.

Details, tables and figures: `report/courtflow_gs_report.pdf`. Scripts behind every measurement:
`experiments/pro6000/`. Local copies of all videos and results: `../pro6000_results/`.

## Running on a fresh machine

```bash
python -m ring_init.doctor --compile     # builds fused MLS, fused SSIM and gsplat; must pass first
python -m ring_init.e2e --scene basketball --dataset-dir <capture> \
    --config ring_init/configs/config.local.json --out-dir <out> --frames 0:700
```

For native 4K use `ring_init/configs/config.4k.example.json` as the local config. To try another
Gaussian budget without restarting, branch from a saved checkpoint:

```bash
python -m ring_init.run --scene basketball --stage b --frames 0:700 --config <out>/basketball/e2e_config.json \
    --tag new_run --resume-from stage_b_v2:250 --set v2_max_gaussians=3000000
```

## Environment notes (Brev instance)

- Run every command through `experiments/pro6000/cfenv.sh`: it puts the conda `libstdc++` first
  (otherwise `CXXABI_1.3.15` import errors) and sets `CUDA_HOME`/`CPATH` so gsplat can JIT-compile.
- `brev exec` hangs unless stdin is `/dev/null`; launch long jobs with `setsid nohup ... &` and a
  `flock` guard, because a dropped SSH connection makes `brev` re-run the command.

## Open items

- Re-export `report/figures/courtflow_gs_pipeline.pdf` from the updated `.drawio` source.
- Main weakness: player tracking in crowded late plays (training mask overlap about 0.16 after
  frame 600; 205 recoveries).
