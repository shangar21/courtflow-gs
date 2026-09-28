"""Time GPU SH colour + rasterization of a baked Gaussian state (no encode/copy), CUDA-synchronized.

    python bench_render.py --config <e2e_config.json> [--ply X.ply] --widths 937,1920,3840 --json-out out.json
Without --ply, the latest Stage-B checkpoint of the configured run is used.
"""
import argparse, json, time
from pathlib import Path
import torch, torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform.track import eval_colors
from ring_init.eval.heldout import load_all_cameras
from ring_init.gs.export import load_ply
from ring_init.render_baked import _checkpoint
from ring_init.render_video import orbit_cameras
from ring_init.stage_a import Scene
from ring_init.stage_b import _camera_frame

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--ply")
ap.add_argument("--widths", default="937,1920,3840"); ap.add_argument("--iters", type=int, default=300); ap.add_argument("--json-out")
args = ap.parse_args(); cfg = Config.load(args.config); scene = Scene("basketball", cfg, set())
if args.ply: frame, path = None, Path(args.ply)
else:
    frame, state = _checkpoint(scene.out / "stage_b_v2", None); path = state / "point_cloud.ply"
model = load_ply(path, cfg.device); p = model.params
means, quats, scales, op = p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"])
sh1 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
cam13 = load_all_cameras(cfg.eval_sparse)[13]
orbit = orbit_cameras(scene.cameras, args.iters, 360.0, 0.0)
results = {"ply": str(path), "checkpoint_frame": frame, "gaussians": len(model), "ply_mb": path.stat().st_size / 1e6,
           "gpu": torch.cuda.get_device_name(), "eval_camera": [cam13.width, cam13.height], "runs": []}
for target_w in (int(w) for w in args.widths.split(",")):
    for kind in ("view13", "orbit"):
        cams = [cam13] * args.iters if kind == "view13" else orbit
        frames = [_camera_frame(c, None, cfg.device, target_w / c.width) for c in cams]
        def one(cf):
            colors = eval_colors(p["sh0"], sh1, means, cf.center)
            return rasterization(means, quats, scales, op, colors, cf.viewmat[None], cf.K[None], cf.width, cf.height, sh_degree=None)[0]
        with torch.no_grad():
            for cf in frames[:20]: one(cf)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
            for cf in frames: one(cf)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) / len(frames)
        r = {"camera": kind, "width": frames[0].width, "height": frames[0].height, "ms": dt * 1e3, "fps": 1 / dt,
             "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30}
        results["runs"].append(r); print(json.dumps(r), flush=True)
if args.json_out: Path(args.json_out).write_text(json.dumps(results, indent=2) + "\n")
