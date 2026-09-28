"""Score several runs' saved checkpoints on the SAME test cameras, resolution and ground truth.

    python fair_compare.py --eval-config <1080p e2e_config.json> --run NAME=OUTROOT_SCENE_DIR/TAG ... --out result.json
All runs share the calibration's world frame, so any checkpoint can be drawn from the 1080p test cameras.
"""
import argparse, json
from pathlib import Path
import cv2, lpips, numpy as np, torch, torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform.track import eval_colors
from ring_init.eval.heldout import load_all_cameras
from ring_init.gs.export import load_ply
from ring_init.gs.train import ssim
from ring_init.stage_a import Scene
from ring_init.stage_b import _camera_frame, source_view

ap = argparse.ArgumentParser(); ap.add_argument("--eval-config", required=True); ap.add_argument("--run", action="append", required=True)
ap.add_argument("--frames", default=",".join(str(f) for f in range(50, 700, 50))); ap.add_argument("--out", required=True)
a = ap.parse_args(); cfg = Config.load(a.eval_config); s = Scene("basketball", cfg, set()); dev = cfg.device
allc = load_all_cameras(cfg.eval_sparse); train = {source_view(c) for c in s.cameras}; held = sorted(v for v in allc if v not in train)
cams = {v: _camera_frame(allc[v], None, dev, 1.0) for v in held}; lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
frames = [int(x) for x in a.frames.split(",")]
gt = {}
res = {}
for spec in a.run:
    name, path = spec.split("=", 1); rows = []
    for f in frames:
        model = load_ply(Path(path) / "ply" / f"frame_{f:06d}" / "point_cloud.ply", dev); p = model.params
        means, quats, scales, op = p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"])
        sh1 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
        ps, ss, ls = [], [], []
        for v in held:
            key = (v, f)
            if key not in gt:
                im = cv2.imread(str(s.out / "eval_frames" / f"view_{v:03d}" / f"{f:06d}.png"))
                gt[key] = torch.as_tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), device=dev).float().div(255).permute(2, 0, 1)
            cf = cams[v]; T = gt[key]
            with torch.no_grad():
                rgb = rasterization(means, quats, scales, op, eval_colors(p["sh0"], sh1, means, cf.center), cf.viewmat[None], cf.K[None], cf.width, cf.height, sh_degree=None)[0][0]
                P = rgb.clamp(0, 1).permute(2, 0, 1)
                ps.append(float(-10 * torch.log10(((P - T) ** 2).mean()))); ss.append(float(ssim(P, T))); ls.append(float(lp(P[None] * 2 - 1, T[None] * 2 - 1).mean()))
        rows.append({"frame": f, "gaussians": len(model), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(ls))})
        print(name, rows[-1], flush=True); del model; torch.cuda.empty_cache()
    res[name] = {"frames": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}}
Path(a.out).write_text(json.dumps({"eval": "1920x1080 test cameras, 24 views", "frames": frames, "runs": res}, indent=2) + "\n")
print(json.dumps({k: v["mean"] for k, v in res.items()}, indent=1))
