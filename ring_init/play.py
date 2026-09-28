"""Forward-only playback of an exported CourtFlow-GS package.

    python -m ring_init.play --package playback --camera view:13 --frames 0:700 --out view13.mp4
    python -m ring_init.play --package playback --camera view:13 --benchmark

The package contains only Gaussian drawing parameters and camera calibration.  It deliberately
does not import SAM2, MASt3R, an optimiser, or the tracker.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def _range(value: str, end: int) -> range:
    a, sep, b = value.partition(":")
    if not sep:
        return range(int(a), int(a) + 1)
    return range(int(a or 0), int(b or end))


def _camera(meta: dict, value: str, device: str, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if not value.startswith("view:"):
        raise ValueError("playback packages currently support physical cameras: use --camera view:<id>")
    view = int(value.split(":", 1)[1])
    raw = next((c for c in meta["cameras"] if int(c["view"]) == view), None)
    if raw is None:
        raise ValueError(f"camera view:{view} is not in this package")
    K = np.asarray(raw["K"], np.float32)
    R, t = np.asarray(raw["R"], np.float32), np.asarray(raw["t"], np.float32)
    w, h = int(raw["width"] * scale) // 2 * 2, int(raw["height"] * scale) // 2 * 2
    K[:2, :2] *= scale
    K[:2, 2] = (K[:2, 2] + .5) * scale - .5
    vm = np.eye(4, dtype=np.float32); vm[:3] = np.column_stack((R, t))
    return (torch.as_tensor(vm, device=device), torch.as_tensor(K, device=device),
            torch.as_tensor(-R.T @ t, device=device), w, h)


def _load(path: Path, device: str) -> dict[str, torch.Tensor]:
    with np.load(path) as raw:
        return {k: torch.as_tensor(raw[k], device=device, dtype=torch.float32 if k != "instance_ids" else torch.int32)
                for k in raw.files}


def _render(parts: list[dict[str, torch.Tensor]], cam: tuple, packed: bool) -> torch.Tensor:
    vm, K, center, w, h = cam
    p = {k: torch.cat([x[k] for x in parts]) for k in parts[0]}
    means = p["means"]; quats = F.normalize(p["quats"], dim=-1)
    direction = F.normalize(means - center, dim=-1)
    x, y, z = direction.unbind(-1)
    color = SH_C0 * p["sh0"][:, 0] + SH_C1 * (-y[:, None] * p["sh1"][:, 0] + z[:, None] * p["sh1"][:, 1] - x[:, None] * p["sh1"][:, 2])
    rgb, _, _ = rasterization(means, quats, torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                              (color + .5).clamp_min(0), vm[None], K[None], w, h, sh_degree=None, packed=packed)
    return rgb[0].clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--camera", default="view:13")
    ap.add_argument("--frames", default="0:")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--packed", action="store_true")
    args = ap.parse_args()
    root = args.package; meta = json.loads((root / "meta.json").read_text())
    frames = _range(args.frames, int(meta["end_frame"]) + 1)
    selected = list(frames)
    if not selected:
        raise ValueError("--frames selects no frames")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Gaussian playback requires a CUDA-enabled PyTorch/gsplat installation")
    cam = _camera(meta, args.camera, device, args.scale)
    bg_files = sorted((int(p.stem.split("_")[-1]), p) for p in root.glob("background_*.npz"))
    if not bg_files:
        raise FileNotFoundError("package has no background_*.npz files")
    if not args.benchmark and args.out is None:
        ap.error("--out is required unless --benchmark is used")
    proc = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                 f"{cam[3]}x{cam[4]}", "-r", str(meta.get("fps", 25)), "-i", "-",
                                 "-c:v", args.codec, "-pix_fmt", "yuv420p", str(args.out)], stdin=subprocess.PIPE)
    loaded_bg, loaded_at = None, None
    load_s = draw_s = encode_s = 0.0
    with torch.inference_mode():
        for frame in selected:
            candidates = [x for x in bg_files if x[0] <= frame]
            if not candidates:
                raise ValueError(f"no background keyframe at or before frame {frame}")
            bg_at, bg_path = candidates[-1]
            t = time.perf_counter()
            if loaded_at != bg_at:
                loaded_bg, loaded_at = _load(bg_path, device), bg_at
            dynamic = _load(root / f"players_{frame:06d}.npz", device)
            load_s += time.perf_counter() - t
            torch.cuda.synchronize(); t = time.perf_counter()
            image = _render([loaded_bg, dynamic], cam, args.packed)
            torch.cuda.synchronize(); draw_s += time.perf_counter() - t
            if proc:
                t = time.perf_counter()
                proc.stdin.write((image.cpu().numpy() * 255 + .5).astype(np.uint8).tobytes())
                encode_s += time.perf_counter() - t
    if proc:
        proc.stdin.close(); proc.wait()
        if proc.returncode:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode})")
    n = len(selected)
    print(f"{n} frames: load {load_s/n*1e3:.2f} ms, draw {draw_s/n*1e3:.2f} ms, encode {encode_s/n*1e3:.2f} ms")


if __name__ == "__main__":
    main()
