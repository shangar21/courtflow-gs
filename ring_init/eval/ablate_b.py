"""Stage B ablations on a short clip (held-out cameras evaluated every eval_frame_stride frames).

    python -m ring_init.eval.ablate_b --scene basketball --config ring_init/configs/basketball.json --frames 0:60
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from ring_init.config import Config

ABLATIONS: dict[str, dict] = {
    "default": {},
    "no_arap": {"arap_weight": 0.0},
    "no_temporal": {"temporal_weight": 0.0},
    "no_warm_start": {"tracking_warm_start": False},
    "K4": {"gaussian_control_knn": 4},
    "K16": {"gaussian_control_knn": 16},
    "cp256": {"control_points": 256},
    "cp1024": {"control_points": 1024},
    "cams2": {"cameras_per_iteration": 2},
    "cams12": {"cameras_per_iteration": 12},
    "cp_rotation_blend": {"mls_blend_cp_rotation": True, "mls_rotation_blend_weight": 0.5},
}


def main() -> None:
    from ring_init.stage_a import Scene
    from ring_init.stage_b import step_track
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--frames", default="0:60"); ap.add_argument("--only", nargs="*")
    args = ap.parse_args(); base = Config.load(args.config); a, _, b = args.frames.partition(":")
    out = None
    for name, override in ABLATIONS.items():
        if args.only and name not in args.only: continue
        s = Scene(args.scene, replace(base, **override)); t = time.time()
        summary = step_track(s, int(a), int(b), tag=f"stage_b_ablation_{name}")
        out = s.out / "stage_b_ablations.json"; prev = json.loads(out.read_text()) if out.is_file() else {}
        prev[name] = {**summary, "wall_s": time.time() - t}; out.write_text(json.dumps(prev, indent=2) + "\n")
    rows = json.loads(out.read_text())
    table = ["| ablation | held-out PSNR | held-out person PSNR | SSIM | LPIPS | train IoU | s/frame (track) | iterations |", "|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        table.append(f"| {name} | {r.get('eval_heldout_psnr', float('nan')):.2f} | {r.get('eval_heldout_person_psnr', float('nan')):.2f} | {r.get('eval_heldout_ssim', float('nan')):.3f} | "
                     f"{r.get('eval_heldout_lpips', float('nan')):.3f} | {r.get('eval_train_iou', float('nan')):.3f} | {r['track_s']:.2f} | {r['iterations']:.1f} |")
    (out.parent / "stage_b_ablations.md").write_text("\n".join(table) + "\n"); print("\n".join(table))


if __name__ == "__main__":
    main()
