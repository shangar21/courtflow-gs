"""Per-player PSNR on the 24 test cameras, comparing runs on identical pixels.

For each saved full-state checkpoint (which keeps each Gaussian's player id), every run draws each
player's footprint; a player's region is the union of all runs' footprints (dilated 6 px), so every
run is scored on the same pixels. PSNR is then pooled over views and frames per player.

    python per_player.py --config <1080p e2e_config.json> --tags stage_b_cap2m,stage_b_cap3m --frames 300,350,...,650 --out pp.json
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform.track import eval_colors
from ring_init.eval.heldout import load_all_cameras
from ring_init.stage_a import Scene
from ring_init.stage_b import _camera_frame, source_view

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--tags", required=True)
ap.add_argument("--frames", default="300,350,400,450,500,550,600,650"); ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args(); cfg = Config.load(a.config); s = Scene("basketball", cfg, set()); dev = cfg.device
tags = a.tags.split(","); frames = [int(x) for x in a.frames.split(",")]
allc = load_all_cameras(cfg.eval_sparse); train = {source_view(c) for c in s.cameras}; held = sorted(v for v in allc if v not in train)
cams = {v: _camera_frame(allc[v], None, dev, 1.0) for v in held}
kernel = torch.ones(1, 1, 13, 13, device=dev)

def load(tag, f):
    p = {k: v.to(dev) for k, v in torch.load(s.out / tag / "state" / f"frame_{f:06d}.pt", map_location="cpu", weights_only=False)["tracker"]["params"].items()}
    ids = p["instance_ids"].long(); sh1 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
    return p, ids, sh1

sse = {t: {} for t in tags}; npx = {}; viewmean = {}
for f in frames:
    models = {t: load(t, f) for t in tags}
    players = sorted(set().union(*[set(m[1].unique().tolist()) for m in models.values()]) - {0})
    for v in held:
        gt = torch.as_tensor(cv2.cvtColor(cv2.imread(str(s.out / "eval_frames" / f"view_{v:03d}" / f"{f:06d}.png")), cv2.COLOR_BGR2RGB), device=dev).float().div(255)
        cf = cams[v]; rgbs, masks = {}, {}
        for t, (p, ids, sh1) in models.items():
            means, quats, scales, op = p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"])
            onehot = (ids[:, None] == torch.as_tensor(players, device=dev)[None]).float()
            feats = torch.cat([eval_colors(p["sh0"], sh1, means, cf.center), onehot], 1)
            with torch.no_grad():
                out = rasterization(means, quats, scales, op, feats, cf.viewmat[None], cf.K[None], cf.width, cf.height, sh_degree=None)[0][0]
            rgbs[t], masks[t] = out[..., :3].clamp(0, 1), out[..., 3:] > a.thr
        dil = lambda m: F.conv2d(m.permute(2, 0, 1)[:, None].float(), kernel, padding=6)[:, 0] > 0       # [P,H,W], dilate 6 px
        st = torch.stack(list(masks.values()))
        regions = {"union": dil(st.any(0)), "intersection": dil(st.all(0))} | {f"own:{t}": dil(masks[t]) for t in tags}
        for t in tags:   # tracker-style: one PSNR per view over all of this run's players, averaged later
            roi = dil(masks[t]).any(0)
            if bool(roi.any()): viewmean.setdefault(t, []).append(float(-10 * torch.log10(((rgbs[t] - gt) ** 2)[roi].mean())))
        for mode, region in regions.items():
            for j, k in enumerate(players):
                n = int(region[j].sum())
                if n < 200: continue
                key = (mode, k); npx[key] = npx.get(key, 0) + n
                for t in tags:
                    if mode.startswith("own:") and mode != f"own:{t}": continue
                    sse[t][key] = sse[t].get(key, 0.0) + float(((rgbs[t] - gt) ** 2)[region[j]].sum() / 3)
    print("frame", f, "done", flush=True)
res = {"frames": frames, "modes": {}}
for mode in ["union", "intersection"] + [f"own:{t}" for t in tags]:
    keys = [key for key in npx if key[0] == mode]
    per = {}
    for key in sorted(keys, key=lambda x: x[1]):
        per[key[1]] = {t: float(-10 * np.log10(sse[t][key] / npx[key])) for t in tags if key in sse[t]} | {"pixels": npx[key]}
    pooled = {t: float(-10 * np.log10(sum(sse[t][k] for k in keys if k in sse[t]) / sum(npx[k] for k in keys if k in sse[t]))) for t in tags if any(k in sse[t] for k in keys)}
    res["modes"][mode] = {"players": per, "pooled": pooled}
    if mode in ("union", "intersection"):
        wins = sum(1 for r in per.values() if r[tags[-1]] > r[tags[0]])
        print(f"{mode:13s} pooled " + " ".join(f"{t}={v:.2f}" for t, v in pooled.items()) + f"  | {tags[-1]} better for {wins}/{len(per)} players")
    else:
        print(f"{mode:28s} pooled " + " ".join(f"{t}={v:.2f} ({sum(r['pixels'] for r in per.values())/1e6:.1f} Mpx)" for t, v in pooled.items()))
res["tracker_style_view_mean"] = {t: float(np.mean(v)) for t, v in viewmean.items()}
print("tracker-style mean of per-view PSNR: " + " ".join(f"{t}={v:.2f}" for t, v in res["tracker_style_view_mean"].items()))
Path(a.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
