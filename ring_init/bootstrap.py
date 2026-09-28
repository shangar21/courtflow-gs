"""Adapt a raw ``cameras/`` + ``calibration/`` capture into CourtFlow-GS inputs.

The raw layout contains 36 synchronized 4K MP4 files and COLMAP text calibration with OPENCV
distortion.  This module creates a reproducible pinhole working set (half resolution by default): the 12
ring training images, all-view pinhole calibration for tracking/evaluation, identity photometric
metadata, and frame-0 images for held-out evaluation.
"""
from __future__ import annotations
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np

from ring_init.io.calib import Camera, load_colmap_text

TRAIN_VIEWS = tuple(range(0, 36, 3))


def _scaled_K(K: np.ndarray, scale: float) -> np.ndarray:
    out = K.copy(); out[:2] *= scale; out[2, 2] = 1.0
    return out


def _pinhole(cam: Camera, scale: float) -> Camera:
    w, h = round(cam.width * scale), round(cam.height * scale)
    Kd = _scaled_K(cam.K, scale)
    K, _ = cv2.getOptimalNewCameraMatrix(Kd, cam.dist, (w, h), 0)
    return Camera(Path(cam.name).with_suffix(".png").name, K, np.zeros(0), cam.R, cam.t, w, h)


def _frame0(video: Path, width: int, height: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:{height}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    if len(raw) != width * height * 3:
        raise RuntimeError(f"{video}: expected one {width}x{height} RGB frame, received {len(raw)} bytes")
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def _write_camera_json(path: Path, cameras: list[Camera]) -> None:
    path.write_text(json.dumps({"units": "meters", "cameras": [
        {"name": c.name, "K": c.K.tolist(), "distortion": c.dist.tolist(), "R": c.R.tolist(), "t": c.t.tolist(), "width": c.width, "height": c.height}
        for c in cameras]}, indent=2) + "\n")


def prepare(dataset_dir: str | Path, work_root: str | Path, scene: str, scale: float = 0.5) -> dict[str, str]:
    """Create/cache the pinhole working set and return paths for a resolved Config."""
    dataset, root = Path(dataset_dir), Path(work_root)
    videos, calib = dataset / "cameras", dataset / "calibration"
    if not videos.is_dir() or not calib.is_dir():
        raise FileNotFoundError(f"Expected {dataset}/cameras and {dataset}/calibration")
    raw = load_colmap_text(calib / "cameras.txt", calib / "images.txt", expected_cameras=36)
    all_pinhole = [_pinhole(c, scale) for c in raw]
    stage = root / "prepared_input" / scene; images, eval_images = stage / "images", stage / "eval_images"
    marker = stage / "prepared.json"
    if marker.is_file() and json.loads(marker.read_text()).get("scale") != scale:
        raise RuntimeError(f"{stage} was prepared at a different scale; use a separate --out-dir per capture scale")
    if not marker.is_file():
        images.mkdir(parents=True, exist_ok=True); eval_images.mkdir(parents=True, exist_ok=True)
        for view, (src, dst) in enumerate(zip(raw, all_pinhole)):
            target = eval_images / f"view_{view:03d}.png"
            if not target.is_file():
                rgb = _frame0(videos / f"view_{view:03d}.mp4", dst.width, dst.height)
                # Distortion coefficients are calibrated at native resolution; after downsampling
                # the scaled intrinsics above make this remap geometrically equivalent.
                Kd = _scaled_K(src.K, scale)
                image = cv2.undistort(rgb, Kd, src.dist, None, dst.K)
                cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if view in TRAIN_VIEWS:
                train = images / f"cam_{TRAIN_VIEWS.index(view):02d}.png"
                if not train.is_file(): train.write_bytes(target.read_bytes())
        train_cams = [all_pinhole[v] for v in TRAIN_VIEWS]
        _write_camera_json(stage / "calibration.json", train_cams)
        _write_camera_json(stage / "all_cameras_undistorted.json", all_pinhole)
        (stage / "photometric_identity.json").write_text(json.dumps({"transforms_to_reference": [
            {"view": v, "gain": [1, 1, 1], "bias": [0, 0, 0]} for v in range(36)]}, indent=2) + "\n")
        marker.write_text(json.dumps({"source": str(dataset.resolve()), "scale": scale, "training_views": TRAIN_VIEWS}, indent=2) + "\n")
    return {"data_root": str(stage.parent), "videos_dir": str(videos), "distorted_cameras_txt": str(calib / "cameras.txt"),
            "distorted_images_txt": str(calib / "images.txt"), "eval_sparse": str(stage / "all_cameras_undistorted.json"),
            "eval_images": str(eval_images), "photometric_json": str(stage / "photometric_identity.json")}
