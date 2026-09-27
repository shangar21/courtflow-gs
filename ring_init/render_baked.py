"""Render a GPU-resident baked Stage-B checkpoint at realtime playback rates.

This is the offline inference path: it performs no SAM2, MLS tracking, or optimisation.  It
loads one of the saved baked PLY states once, keeps it on the GPU, and renders it from a fixed
physical/held-out view or a moving orbit.

    PYTHONPATH=. python -m ring_init.render_baked --scene basketball \
      --config ring_init/configs/basketball.json --tag stage_b_diag_crop_ball \
      --checkpoint 299 --camera orbit --frames 0:300 --out final_orbit.mp4
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

from ring_init.config import Config
from ring_init.deform.track import eval_colors
from ring_init.eval.heldout import load_all_cameras
from ring_init.gs.export import load_ply
from ring_init.render_video import orbit_cameras
from ring_init.stage_a import Scene
from ring_init.stage_b import _camera_frame


def _checkpoint(path: Path, frame: int | None) -> tuple[int, Path]:
    states = sorted((int(p.name.split("_")[-1]), p) for p in (path / "ply").glob("frame_*") if (p / "point_cloud.ply").is_file())
    if not states:
        raise FileNotFoundError(f"No baked PLY checkpoints under {path / 'ply'}")
    if frame is None:
        return states[-1]
    for f, p in states:
        if f == frame:
            return f, p
    raise FileNotFoundError(f"No checkpoint for frame {frame}; available {[f for f, _ in states]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True); ap.add_argument("--config", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", type=int, help="baked frame to load (default: latest)")
    ap.add_argument("--camera", default="view:13", help="view:<0-35> | train:<0-11> | orbit")
    ap.add_argument("--frames", default="0:300", help="number of output frames; source indices only determine count")
    ap.add_argument("--orbit-degrees", type=float, default=360.0); ap.add_argument("--orbit-start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25); ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--out", required=True)
    ap.add_argument("--codec", default="h264_nvenc", help="ffmpeg video codec (default: RTX hardware H.264 encoder; use libx264 as fallback)")
    ap.add_argument("--packed", action="store_true", help="use gsplat's packed inference path (benchmark before deployment)")
    args = ap.parse_args(); cfg = Config.load(args.config); scene = Scene(args.scene, cfg, set()); root = scene.out / args.tag
    frame, state = _checkpoint(root, args.checkpoint); model = load_ply(state / "point_cloud.ply", cfg.device)
    a, _, b = args.frames.partition(":"); count = max(0, int(b) - int(a)) if b else int(a)
    allc = load_all_cameras(cfg.eval_sparse)
    if args.camera == "orbit": cams = orbit_cameras(scene.cameras, count, args.orbit_degrees, args.orbit_start)
    elif args.camera.startswith("train:"): cams = [scene.cameras[int(args.camera.split(":", 1)[1])]] * count
    elif args.camera.startswith("view:"): cams = [allc[int(args.camera.split(":", 1)[1])]] * count
    else: raise ValueError("--camera must be view:<id>, train:<index>, or orbit")
    if not cams: raise ValueError("--frames must select at least one output frame")
    width, height = int(cams[0].width * args.scale) // 2 * 2, int(cams[0].height * args.scale) // 2 * 2
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    quality = ["-cq", "18"] if args.codec.endswith("_nvenc") else ["-crf", "18"]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-",
                             "-c:v", args.codec, "-pix_fmt", "yuv420p", *quality, str(out)], stdin=subprocess.PIPE)
    p = model.params
    with torch.no_grad():
        means, quats, scales, op = p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"])
        sh1 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
        for cam in cams:
            # Render at the requested output resolution rather than rasterizing the native
            # frame and downsampling afterward.  This is the same camera/image scale and avoids
            # paying full-resolution raster cost for a preview or realtime stream.
            cf = _camera_frame(cam, None, cfg.device, args.scale)
            colors = eval_colors(p["sh0"], sh1, means, cf.center)
            rgb, _, _ = rasterization(means, quats, scales, op, colors, cf.viewmat[None], cf.K[None], cf.width, cf.height,
                                      sh_degree=None, packed=args.packed)
            img = F.interpolate(rgb[0].permute(2, 0, 1)[None], size=(height, width), mode="area")[0].permute(1, 2, 0)
            proc.stdin.write((img.clamp(0, 1).cpu().numpy() * 255 + .5).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    if proc.returncode: raise RuntimeError(f"ffmpeg failed ({proc.returncode})")
    print(f"Rendered baked frame {frame} ({len(model):,} Gaussians) to {out} at {args.fps} FPS.")


if __name__ == "__main__":
    main()
