from __future__ import annotations
from pathlib import Path
import numpy as np
import open3d as o3d


def fuse_points(points: np.ndarray, colors: np.ndarray, voxel_m: float, outlier_std_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0: return np.empty((0,3)), np.empty((0,3)), np.empty((0,3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=min(20, len(points)), std_ratio=outlier_std_ratio)
    cloud = cloud.voxel_down_sample(voxel_m)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(2*voxel_m, 1e-4), max_nn=30))
    return np.asarray(cloud.points), np.asarray(cloud.colors), np.asarray(cloud.normals)


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray, normals: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    if normals is not None: cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, float))
    Path(path).parent.mkdir(parents=True, exist_ok=True); o3d.io.write_point_cloud(str(path), cloud)
