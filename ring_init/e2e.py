"""One-command canonical reconstruction, dynamic tracking, evaluation, and final renders.

Use ``--dataset-dir`` for the raw capture layout (``cameras/`` + ``calibration/``); the runner
bootstraps all pinhole inputs itself. ``--train-dir`` remains available for already prepared
Stage-A image datasets.

Example:

    python -m ring_init.e2e --scene basketball --dataset-dir /datasets/basketball_capture \
      --config ring_init/configs/config.local.json --out-dir /outputs/ring_final
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

from ring_init.config import Config
from ring_init.run import stage_a, stage_b


def _publish_animated_renders(run_root: Path, tag: str, eval_view: int) -> None:
    """Publish videos rendered from each live per-frame state during Stage B.

    A single baked PLY is a scene at one instant. Repeating the final PLY would make a
    static video, so E2E publishes Stage B's per-frame held-out and orbit renders instead.
    """
    source = run_root / tag / "videos"
    final = run_root / tag / "final_renders"; final.mkdir(parents=True, exist_ok=True)
    inputs = {
        source / f"heldout_view{eval_view:02d}.mp4": final / f"animated_eval_view{eval_view:02d}.mp4",
        source / "orbit360.mp4": final / "animated_orbit360.mp4",
    }
    for src, dst in inputs.items():
        if not src.is_file():
            raise FileNotFoundError(f"Expected animated Stage-B render was not produced: {src}")
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-dir", help="directory containing <scene>/ pre-extracted training images")
    source.add_argument("--dataset-dir", help="raw capture root containing cameras/ and calibration/")
    ap.add_argument("--config", required=True, help="local JSON with camera/video/SAM2/evaluation paths")
    ap.add_argument("--out-dir", help="output root (default: <train-dir>/ring_dynamic_outputs)")
    ap.add_argument("--frames", default="0:700", help="Stage-B frame range (the complete supplied 28-second sequence)")
    ap.add_argument("--force", default="", help="comma-separated Stage-A/B cache steps to recompute")
    ap.add_argument("--eval-view", type=int, default=13, help="held-out camera for final playback")
    ap.add_argument("--orbit-frames", type=int, default=0,
                    help="deprecated; the final orbit is the animated Stage-B orbit over the requested frame range")
    ap.add_argument("--prepare-only", action="store_true", help="validate/bootstrap raw cameras+calibration, then stop before training")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.rescale_pixel_params()  # no-op at the default half-resolution capture scale
    base = Path(args.dataset_dir or args.train_dir).resolve()
    cfg.out_root = str(Path(args.out_dir).resolve()) if args.out_dir else str(base / "ring_dynamic_outputs")
    if args.dataset_dir:
        from ring_init.bootstrap import prepare
        prepared = prepare(base, cfg.out_root, args.scene, cfg.capture_scale)
        for key, value in prepared.items(): setattr(cfg, key, value)
        cfg.calibration = "calibration.json"
    else:
        cfg.data_root = str(base)
    cfg.stage_b_mode = "scene"  # final QuickCapture-style MLS method; legacy v1 is never selected here.
    # The final held-out video is rendered online during Stage B. Ensure the requested view is
    # included even if a user changed the default diagnostic-video views in their local config.
    cfg.v2_video_views = tuple(sorted(set(cfg.v2_video_views) | {args.eval_view}))
    force = {x for x in args.force.split(",") if x}

    # Persist the exact resolved run configuration with outputs, then use it for render commands.
    run_root = Path(cfg.out_root) / args.scene
    run_root.mkdir(parents=True, exist_ok=True)
    resolved = run_root / "e2e_config.json"; cfg.save(resolved)
    if args.prepare_only:
        print(f"Capture bootstrap complete. Resolved config: {resolved}")
        return
    stage_a(args.scene, cfg, force, None, True)
    stage_b(args.scene, cfg, args.frames, force)

    tag = "stage_b_v2"  # ring_init.run's production scene tracker output.
    _publish_animated_renders(run_root, tag, args.eval_view)
    print(f"E2E run complete. Outputs: {run_root}")


if __name__ == "__main__":
    main()
