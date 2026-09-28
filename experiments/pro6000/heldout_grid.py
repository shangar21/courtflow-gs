"""Qualitative held-out grid: render a saved Stage-B checkpoint for all 24 held-out cameras, score each
against its real image, and show the best / median / worst views as ground truth | render.

    python heldout_grid.py --config <e2e_config.json> --tag stage_b_v2 --frames 100,650 --out grid.png
Views are chosen by this frame's full-image PSNR, so the selection is objective, not hand-picked.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform.track import eval_colors
from ring_init.eval.heldout import load_all_cameras
from ring_init.gs.export import load_ply
from ring_init.stage_a import Scene
from ring_init.stage_b import _camera_frame, source_view

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--tag", default="stage_b_v2")
ap.add_argument("--frames", default="100,650"); ap.add_argument("--out", required=True); ap.add_argument("--thumb", type=int, default=640)
a = ap.parse_args(); cfg = Config.load(a.config); s = Scene("basketball", cfg, set()); dev = cfg.device
allc = load_all_cameras(cfg.eval_sparse); train = {source_view(c) for c in s.cameras}
held = sorted(v for v in allc if v not in train)
rows, report = [], {}
for f in (int(x) for x in a.frames.split(",")):
    model = load_ply(s.out / a.tag / "ply" / f"frame_{f:06d}" / "point_cloud.ply", dev); p = model.params
    means, quats, scales, op = p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"])
    sh1 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
    scored = []
    for v in held:
        gt_path = s.out / "eval_frames" / f"view_{v:03d}" / f"{f:06d}.png"
        gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
        cf = _camera_frame(allc[v], None, dev, 1.0)
        with torch.no_grad():
            rgb = rasterization(means, quats, scales, op, eval_colors(p["sh0"], sh1, means, cf.center), cf.viewmat[None], cf.K[None], cf.width, cf.height, sh_degree=None)[0][0]
        r = (rgb.clamp(0, 1).cpu().numpy() * 255 + .5).astype(np.uint8)
        psnr = float(10 * np.log10(255 ** 2 / np.mean((r.astype(np.float64) - gt) ** 2)))
        scored.append((psnr, v, gt, r))
    scored.sort(key=lambda t: -t[0]); pick = {"best": scored[0], "median": scored[len(scored) // 2], "worst": scored[-1]}
    report[f] = {"per_view_psnr": {int(v): round(ps, 2) for ps, v, _, _ in scored}, "picked": {k: int(t[1]) for k, t in pick.items()}}
    for label, (psnr, v, gt, r) in pick.items():
        h = round(gt.shape[0] * a.thumb / gt.shape[1]); th = lambda im: cv2.resize(im, (a.thumb, h), interpolation=cv2.INTER_AREA)
        row = np.concatenate([th(gt), np.full((h, 6, 3), 255, np.uint8), th(r)], 1)
        cv2.putText(row, f"frame {f}, test view {v} ({label}), render PSNR {psnr:.1f} dB", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(row); rows.append(np.full((6, row.shape[1], 3), 255, np.uint8))
    del model; torch.cuda.empty_cache()
cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(rows[:-1], 0), cv2.COLOR_RGB2BGR))
Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps({f: r["picked"] for f, r in report.items()}))
