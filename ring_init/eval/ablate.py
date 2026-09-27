"""Stage A ablations on the leave-one-camera-out development metric (person models only).

    python -m ring_init.eval.ablate --scene basketball --config ring_init/configs/basketball.json --camera 4
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from ring_init.config import Config

ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_hull": {"use_hull": False},
    "no_dropgaussian": {"drop_gaussian": False},
    "no_mask_loss": {"alpha_weight": 0.0},
    "no_depth_loss": {"depth_weight": 0.0},
    "sh0": {"sh_degree": 0},
    "dropgaussian_0.2": {"drop_gaussian_rate": 0.2},
    "iters_3000": {"t0_iterations": 3000},
}


def main() -> None:
    from ring_init.eval.loo import run_loo
    from ring_init.stage_a import Scene
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--camera", type=int, nargs="+", default=[4]); ap.add_argument("--only", nargs="*")
    args = ap.parse_args(); base = Config.load(args.config); rows = {}
    for name, override in ABLATIONS.items():
        if args.only and name not in args.only: continue
        s = Scene(args.scene, replace(base, **override)); t = time.time()
        result = run_loo(s, tuple(args.camera), tag_prefix=f"ablation_{name}")
        rows[name] = {**result["mean"], "seconds": time.time() - t}
        out = s.out / "ablations.json"; prev = json.loads(out.read_text()) if out.is_file() else {}
        prev[name] = rows[name]; out.write_text(json.dumps(prev, indent=2) + "\n")
    table = ["| ablation | person PSNR | person SSIM | person LPIPS | alpha IoU | wall s |", "|---|---|---|---|---|---|"]
    for name, r in json.loads((s.out / "ablations.json").read_text()).items():
        table.append(f"| {name} | {r['psnr_person']:.2f} | {r['ssim_person_bbox']:.3f} | {r['lpips_person_bbox']:.3f} | {r['iou']:.3f} | {r['seconds']:.0f} |")
    (s.out / "ablations.md").write_text("\n".join(table) + "\n"); print("\n".join(table))


if __name__ == "__main__":
    main()
