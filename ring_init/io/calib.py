from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray  # world-to-camera
    t: np.ndarray  # world-to-camera
    width: int
    height: int

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.column_stack((self.R, self.t.reshape(3)))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        cam = (self.R @ xyz.T).T + self.t
        uvw = (self.K @ cam.T).T
        return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12), cam[:, 2]


def _camera(value: dict, index: int) -> Camera:
    required = ("K", "R", "t", "width", "height")
    missing = [k for k in required if k not in value]
    if missing:
        raise ValueError(f"camera {index} lacks {missing}")
    K = np.asarray(value["K"], dtype=np.float64).reshape(3, 3)
    R = np.asarray(value["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(value["t"], dtype=np.float64).reshape(3)
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
        raise ValueError(f"camera {index}: R must be a proper world-to-camera rotation")
    return Camera(str(value.get("name", f"cam_{index:02d}")), K,
                  np.asarray(value.get("distortion", value.get("dist", [])), dtype=np.float64).reshape(-1),
                  R, t, int(value["width"]), int(value["height"]))


def load_json(path: str | Path, expected_cameras: int = 12) -> list[Camera]:
    """Load `{units, cameras:[{K,R,t,width,height,distortion}]}` calibration JSON."""
    payload = json.loads(Path(path).read_text())
    if not payload.get("units"):
        raise ValueError("Calibration JSON must declare non-empty `units`.")
    cameras = [_camera(v, i) for i, v in enumerate(payload.get("cameras", []))]
    if len(cameras) != expected_cameras:
        raise ValueError(f"Expected {expected_cameras} cameras, got {len(cameras)}")
    return cameras


def load_colmap_text(cameras_txt: str | Path, images_txt: str | Path, expected_cameras: int = 12) -> list[Camera]:
    """COLMAP text adapter. It consumes known poses only; it never calls COLMAP SfM."""
    intrinsics: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for line in Path(cameras_txt).read_text().splitlines():
        if not line or line.startswith("#"): continue
        parts = line.split(); cid, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        p = list(map(float, parts[4:]))
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.]])
            dist = np.array(p[3:])
        elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
            K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.]])
            dist = np.array(p[4:])
        else: raise ValueError(f"Unsupported COLMAP camera model {model}")
        intrinsics[cid] = K, dist, w, h
    lines = [x for x in Path(images_txt).read_text().splitlines() if x and not x.startswith("#")]
    result: list[Camera] = []
    for line in lines:
        p = line.split()
        if len(p) < 10: continue  # points2D line
        qw, qx, qy, qz, tx, ty, tz, cid = map(float, p[1:9]); cid = int(cid)
        R, _ = cv2.Rodrigues(np.array([[qx], [qy], [qz]]) * 0)  # overwritten below
        q = np.array([qw, qx, qy, qz], float); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        K, dist, width, height = intrinsics[cid]
        result.append(Camera(p[9], K, dist, R, np.array([tx,ty,tz]), width, height))
    if len(result) != expected_cameras: raise ValueError(f"Expected {expected_cameras} image poses, got {len(result)}")
    return result


def undistort(image: np.ndarray, camera: Camera) -> tuple[np.ndarray, Camera]:
    if camera.dist.size == 0 or np.allclose(camera.dist, 0): return image.copy(), camera
    Knew, _ = cv2.getOptimalNewCameraMatrix(camera.K, camera.dist, (camera.width, camera.height), 0)
    output = cv2.undistort(image, camera.K, camera.dist, None, Knew)
    return output, Camera(camera.name, Knew, np.zeros(0), camera.R, camera.t, output.shape[1], output.shape[0])


def save_colmap_text(cameras: list[Camera], out_dir: str | Path) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lines = ["# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for i, c in enumerate(cameras, 1): lines.append(f"{i} PINHOLE {c.width} {c.height} {c.K[0,0]} {c.K[1,1]} {c.K[0,2]} {c.K[1,2]}")
    (out / "cameras.txt").write_text("\n".join(lines) + "\n")
