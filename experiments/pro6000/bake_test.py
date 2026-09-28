"""Controls-only playback: bake frame g from a saved checkpoint at frame f using only the stored
control positions (no per-frame Gaussian updates), score it on the 24 test cameras, and time the bake.

    python bake_test.py --config <1080p e2e_config.json> --tag stage_b_cap2m --from 600 --to 610,620,630,640,650 --out bake.json
"""
import argparse, json, time
from pathlib import Path
import cv2, lpips, numpy as np, torch, torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform import mls
from ring_init.deform.track import eval_colors
from ring_init.eval.heldout import load_all_cameras
from ring_init.gs.train import ssim
from ring_init.stage_a import Scene
from ring_init.stage_b import _camera_frame, source_view

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--from", dest="src", type=int, required=True); ap.add_argument("--to", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); cfg = Config.load(a.config); s = Scene("basketball", cfg, set()); dev = cfg.device
run = s.out / a.tag
st = torch.load(run / "state" / f"frame_{a.src:06d}.pt", map_location="cpu", weights_only=False)
p = {k: v.to(dev) for k, v in st["tracker"]["params"].items()}
rest, nbr, w = (st["tracker"]["ctrl"][k].to(dev) for k in ("pos", "nbr", "w"))
P = torch.as_tensor(np.load(run / "control_positions.npz")["pos"], device=dev)
means0, quats0 = p["means"], F.normalize(p["quats"], dim=-1)
sh1_0 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
scales, op, sh0 = torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), p["sh0"]
ids = p["instance_ids"].long()
allc = load_all_cameras(cfg.eval_sparse); train = {source_view(c) for c in s.cameras}; held = sorted(v for v in allc if v not in train)
cams = {v: _camera_frame(allc[v], None, dev, 1.0) for v in held}; lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
recorded = {x["frame"]: x["eval"] for x in json.loads((run / "metrics.json").read_text())["frames"] if "eval" in x and "heldout_psnr" in x["eval"]}

def score(means, quats, sh1, f):
    ps, ss, ls = [], [], []
    for v in held:
        T = torch.as_tensor(cv2.cvtColor(cv2.imread(str(s.out / "eval_frames" / f"view_{v:03d}" / f"{f:06d}.png")), cv2.COLOR_BGR2RGB), device=dev).float().div(255).permute(2, 0, 1)
        cf = cams[v]
        with torch.no_grad():
            P_ = rasterization(means, quats, scales, op, eval_colors(sh0, sh1, means, cf.center), cf.viewmat[None], cf.K[None], cf.width, cf.height, sh_degree=None)[0][0].clamp(0, 1).permute(2, 0, 1)
            ps.append(float(-10 * torch.log10(((P_ - T) ** 2).mean()))); ss.append(float(ssim(P_, T))); ls.append(float(lp(P_[None] * 2 - 1, T[None] * 2 - 1).mean()))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(ls))}

out = {"source_frame": a.src, "gaussians": int(len(means0)), "dynamic_gaussians": int((ids != 0).sum()), "controls": int(len(rest)), "rows": []}
out["checkpoint_itself"] = {**score(means0, quats0, sh1_0, a.src), "recorded_by_tracker": recorded.get(a.src, {}).get("heldout_psnr")}
for g in (int(x) for x in a.to.split(",")):
    t = (P[g] - P[a.src]).float().contiguous()
    with torch.no_grad():
        for _ in range(3): mls.deform(means0, quats0, sh1_0, rest, nbr, w, t)          # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): m, q, sh = mls.deform(means0, quats0, sh1_0, rest, nbr, w, t)
        torch.cuda.synchronize(); bake_ms = (time.perf_counter() - t0) / 20 * 1e3
    row = {"frame": g, "frames_since_checkpoint": g - a.src, "bake_ms": bake_ms,
           "baked": score(m, F.normalize(q, dim=-1), sh, g), "not_moved": score(means0, quats0, sh1_0, g),
           "full_run_recorded": recorded.get(g, {}).get("heldout_psnr")}
    out["rows"].append(row); print(json.dumps(row), flush=True)
Path(a.out).write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
