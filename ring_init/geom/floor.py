"""Floor plane, scene scale, and court volume from full-frame known-pose points."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from ring_init.io.calib import Camera


@dataclass
class Floor:
    normal: np.ndarray        # unit, pointing up (towards the cameras)
    offset: float             # height(x) = normal @ x + offset
    origin: np.ndarray        # a point on the floor
    axis_u: np.ndarray        # in-plane basis
    axis_v: np.ndarray
    extent_lo: np.ndarray     # (u, v) robust floor extent
    extent_hi: np.ndarray
    units_per_meter: float | None = None

    def height(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz) @ self.normal + self.offset

    def to_plane(self, xyz: np.ndarray) -> np.ndarray:
        d = np.asarray(xyz) - self.origin
        return np.stack((d @ self.axis_u, d @ self.axis_v), -1)

    def from_plane(self, uv: np.ndarray, h: np.ndarray | float = 0.0) -> np.ndarray:
        uv = np.asarray(uv, float)
        return self.origin + uv[..., :1]*self.axis_u + uv[..., 1:2]*self.axis_v + np.asarray(h)[..., None]*self.normal

    def ray_hit(self, camera: Camera, uv: np.ndarray) -> np.ndarray:
        """Intersect pixel rays with the floor plane."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        rays = (camera.R.T @ (np.linalg.inv(camera.K) @ np.c_[uv, np.ones(len(uv))].T)).T
        s = -(self.height(camera.center)) / (rays @ self.normal)
        return camera.center + s[:, None]*rays

    def save(self, path: str | Path) -> None:
        payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Floor":
        p = json.loads(Path(path).read_text())
        return cls(**{k: (np.asarray(v) if isinstance(v, list) else v) for k, v in p.items()})


def fit_floor(points: np.ndarray, cameras: list[Camera], threshold: float, iterations: int, extent_percentile: float, seed: int = 0) -> Floor:
    import open3d as o3d
    o3d.utility.random.seed(seed)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    plane, inliers = cloud.segment_plane(threshold, 3, iterations)
    n = np.asarray(plane[:3], float); d = float(plane[3]); scale = np.linalg.norm(n); n, d = n/scale, d/scale
    centers = np.stack([c.center for c in cameras])
    if np.median(centers @ n + d) < 0: n, d = -n, -d
    if (centers @ n + d).min() <= 0: raise RuntimeError("Floor fit invalid: some camera lies below the floor plane.")
    floor_pts = np.asarray(points)[inliers]
    origin = floor_pts.mean(0); origin = origin - (origin @ n + d)*n
    helper = np.array([1., 0, 0]) if abs(n[0]) < .9 else np.array([0, 0, 1.])
    u = np.cross(n, helper); u /= np.linalg.norm(u); v = np.cross(n, u)
    # Align u with the principal floor direction so the court box is tight.
    uv = np.stack(((floor_pts-origin) @ u, (floor_pts-origin) @ v), 1)
    evals, evecs = np.linalg.eigh(np.cov(uv.T)); major = evecs[:, -1]
    u, v = major[0]*u + major[1]*v, -major[1]*u + major[0]*v
    uv = np.stack(((floor_pts-origin) @ u, (floor_pts-origin) @ v), 1)
    lo, hi = np.percentile(uv, extent_percentile, 0), np.percentile(uv, 100-extent_percentile, 0)
    return Floor(n, d, origin, u, v, lo, hi)


def estimate_units_per_meter(floor: Floor, cameras: list[Camera], masks_per_camera: list[list[np.ndarray]], person_height_m: float) -> float:
    """Median standing height of detected people (feet on the floor) over the metric prior."""
    heights = []
    for cam, masks in zip(cameras, masks_per_camera):
        for m in masks:
            ys, xs = np.nonzero(m)
            if len(ys) < 50: continue
            yb, yt = ys.max(), ys.min()
            foot = floor.ray_hit(cam, np.array([[xs[ys >= yb-2].mean(), yb]]))[0]
            fp = floor.to_plane(foot)
            if np.any(fp < floor.extent_lo) or np.any(fp > floor.extent_hi): continue  # stands, not court
            top = np.array([xs[ys <= yt+2].mean(), yt])
            ray = cam.R.T @ np.linalg.inv(cam.K) @ np.r_[top, 1.]; ray /= np.linalg.norm(ray)
            n = floor.normal; C = cam.center
            A = np.array([[1., -(n @ ray)], [n @ ray, -1.]]); b = np.array([(C-foot) @ n, (C-foot) @ ray])
            h = np.linalg.solve(A, b)[0]
            if h > 0: heights.append(h)
    if len(heights) < 10: raise RuntimeError(f"Only {len(heights)} person heights; cannot estimate scene scale.")
    # Upper quartile: people are often bent; standing players dominate the top of the distribution.
    return float(np.percentile(heights, 75) / person_height_m)
