"""Play back a tracked sequence from any ring camera or a user-defined orbit to an mp4.

    python -m ring_init.render_video --scene basketball --config ring_init/configs/basketball.json \
        --camera view:13 --frames 0:300 --out out.mp4
    python -m ring_init.render_video ... --camera orbit --orbit-degrees 90 --frames 0:300
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from ring_init.config import Config
from ring_init.io.calib import Camera


def orbit_cameras(ring: list[Camera], count: int, degrees: float, start_deg: float = 0.0) -> list[Camera]:
    """Virtual cameras moving along the physical rig: centres linearly and rotations spherically
    interpolated between consecutive ring cameras (ordered by angle), so the path never leaves
    the region the capture cameras constrain. `degrees` of 360 covers the whole ring."""
    from scipy.spatial.transform import Rotation, Slerp
    centers = np.stack([c.center for c in ring]); mid = centers.mean(0)
    up = np.linalg.svd(centers - mid)[2][-1]; ref = centers[0] - mid; ref -= (ref @ up) * up; side = np.cross(up, ref)
    ang = np.degrees(np.arctan2((centers - mid) @ side, (centers - mid) @ ref)) % 360
    order = np.argsort(ang); cams = [ring[i] for i in order]; ang = ang[order]
    out = []
    for i in range(count):
        a = (ang[0] + start_deg + degrees * i / max(count - 1, 1)) % 360
        j = int(np.searchsorted(ang, a, side="right") - 1) % len(cams); k = (j + 1) % len(cams)
        span = (ang[k] - ang[j]) % 360 or 360; u = ((a - ang[j]) % 360) / span
        c = (1 - u) * cams[j].center + u * cams[k].center
        R = Slerp([0, 1], Rotation.from_matrix([cams[j].R, cams[k].R]))(u).as_matrix()
        K = (1 - u) * cams[j].K + u * cams[k].K; base = cams[j]
        out.append(Camera(f"orbit_{i}", K, np.zeros(0), R, -R @ c, base.width, base.height))
    return out


def main() -> None:
    from ring_init.deform.track import Persons, render_persons
    from ring_init.deform import mls
    from ring_init.eval.heldout import load_all_cameras
    from ring_init.stage_a import Scene
    from ring_init.stage_b import _camera_frame, load_canonical, near_plane
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--camera", default="view:13", help="view:<0-35> | train:<0-11> | orbit")
    ap.add_argument("--orbit-degrees", type=float, default=90.0); ap.add_argument("--orbit-start", type=float, default=0.0)
    ap.add_argument("--frames", default="0:300"); ap.add_argument("--tag", default="stage_b"); ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--out", required=True)
    ap.add_argument("--no-near-plane", action="store_true")
    ap.add_argument("--hold", type=int, default=1, help="repeat each selected frame N times (e.g. orbit around a frozen frame)")
    args = ap.parse_args(); cfg = Config.load(args.config); s = Scene(args.scene, cfg); dev = cfg.device
    models = load_canonical(s); persons = Persons.from_models(models)
    st = np.load(s.out / args.tag / "cp_states.npz"); topo = torch.load(s.out / args.tag / "topology.pt")
    frames = st["frames"].tolist(); a, _, b = args.frames.partition(":"); lo, hi = int(a), int(b) if b else frames[-1] + 1
    sel = [i for i, f in enumerate(frames) if lo <= f < hi for _ in range(args.hold)]
    if args.camera == "orbit": cams = orbit_cameras(s.cameras, len(sel), args.orbit_degrees, args.orbit_start)
    elif args.camera.startswith("train:"): cams = [s.cameras[int(args.camera.split(":")[1])]] * len(sel)
    else: cams = [load_all_cameras(cfg.eval_sparse)[int(args.camera.split(":")[1])]] * len(sel)
    W, H = int(cams[0].width * args.scale) // 2 * 2, int(cams[0].height * args.scale) // 2 * 2
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out], stdin=subprocess.PIPE)
    frame_cache = {}; near = 0.0 if args.no_near_plane else near_plane(s)
    print(f"near plane {near:.3f} scene units")
    with torch.no_grad():
        for n, i in enumerate(sel):
            cam = cams[n]; key = cam.name
            if key not in frame_cache or args.camera == "orbit": frame_cache = {key: _camera_frame(cam, models.get(0), dev, 1.0, near)}
            cf = frame_cache[key]
            t = torch.as_tensor(st["t"][i], device=dev); q = torch.as_tensor(st["q"][i], device=dev)
            means, quats, sh1 = mls.deform(persons.means, persons.quats, persons.sh1, topo["rest"], topo["gaussian_neighbors"], topo["gaussian_weights"], t, q,
                                           cfg.mls_blend_cp_rotation, cfg.mls_rotation_blend_weight, cfg.mls_svd_epsilon)
            rgb, _, alpha = render_persons(persons, means, quats, sh1, cf)
            img = (rgb + (1 - alpha[..., None]) * cf.background).clamp(0, 1)
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(H, W), mode="area")[0].permute(1, 2, 0)
            proc.stdin.write((img.cpu().numpy() * 255 + .5).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait(); print(f"wrote {args.out} ({len(sel)} frames)")


if __name__ == "__main__":
    main()
