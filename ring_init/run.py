"""End-to-end CLI.

    PYTHONPATH=.:third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r \
        python -m ring_init.run --scene basketball --stage a --config ring_init/configs/basketball.json

Stage A steps (cached, resumable): calib, match_full, floor, persons, instances, hulls,
match_crops, fuse, train, export, eval. `--force step1,step2` recomputes those steps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ring_init.config import Config
from ring_init.stage_a import Scene, run_geometry, step_export, step_train


def stage_a(scene: str, cfg: Config, force: set[str], until: str | None, evaluate: bool) -> None:
    s = Scene(scene, cfg, force)
    run_geometry(s, until)
    if until in (None, "train", "export", "eval"):
        step_train(s)
        step_export(s)
    if evaluate and until in (None, "eval"):
        from ring_init.eval.heldout import evaluate as run_eval
        run_eval(scene, cfg)


def stage_b(scene: str, cfg: Config, frames: str, force: set[str], exclude: tuple[int, ...] = (), tag: str | None = None,
            resume_from: tuple[str, int] | None = None) -> None:
    """Online tracking of the frozen Stage A canonical models over `frames` (START:END, START = 0)."""
    from ring_init.stage_b import step_frames, step_track, step_track_scene, step_video_masks
    canonical = Path(cfg.out_root) / scene / "canonical" / "manifest.json"
    if not canonical.is_file(): raise FileNotFoundError(f"Stage B requires the Stage A canonical export: {canonical}")
    a, _, b = frames.partition(":"); start, end = int(a), int(b)
    s = Scene(scene, cfg, force)
    step_frames(s, start, end)
    step_video_masks(s, start, end)
    if cfg.stage_b_mode == "scene":
        if cfg.ball:
            from ring_init.ball import step_ball_canonical, track_ball
            if not (s.out / "ball" / "trajectory.npz").is_file() or "ball" in force: track_ball(s, start, end)
            if not (s.out / "canonical_ball" / "manifest.json").is_file() or "ball" in force: step_ball_canonical(s)
        step_track_scene(s, start, end, tag=tag or "stage_b_v2", resume_from=resume_from)
    else: step_track(s, start, end, exclude_cameras=exclude)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True); parser.add_argument("--stage", choices=("a", "b", "both"), required=True)
    parser.add_argument("--frames", default="0:300"); parser.add_argument("--config")
    parser.add_argument("--force", default="", help="comma-separated steps to recompute")
    parser.add_argument("--tag", help="Stage-B output directory name; use this to preserve prior runs")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config field for this run; VALUE may be JSON or a bare string")
    parser.add_argument("--resume-from", metavar="TAG:FRAME", help="Stage B: continue from another run's full checkpoint (use a new --tag)")
    parser.add_argument("--until", default=None, help="stop after this step")
    parser.add_argument("--no-eval", action="store_true", help="skip held-out evaluation")
    args = parser.parse_args(); cfg = Config.load(args.config)
    fields = set(cfg.__dataclass_fields__)
    for item in args.overrides:
        key, sep, raw = item.partition("=")
        if not sep or key not in fields:
            parser.error(f"--set must name a valid config field: {item!r}")
        try: value = json.loads(raw)
        except json.JSONDecodeError: value = raw
        setattr(cfg, key, tuple(value) if key in {"heldout_cameras", "eval_training_views", "v2_video_views"} and isinstance(value, list) else value)
    cfg.validate()
    if args.stage in ("a", "both"): stage_a(args.scene, cfg, {x for x in args.force.split(",") if x}, args.until, not args.no_eval)
    resume = None
    if args.resume_from:
        src, sep, frame = args.resume_from.rpartition(":")
        if not sep or not frame.isdigit(): parser.error("--resume-from must be TAG:FRAME")
        if not args.tag or args.tag == src: parser.error("--resume-from needs a new --tag so the source run is not overwritten")
        resume = (src, int(frame))
    if args.stage in ("b", "both"): stage_b(args.scene, cfg, args.frames, {x for x in args.force.split(",") if x}, tag=args.tag, resume_from=resume)


if __name__ == "__main__":
    main()
