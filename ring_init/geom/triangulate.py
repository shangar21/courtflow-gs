from __future__ import annotations
import numpy as np
from ring_init.io.calib import Camera


def triangulate_dlt(uv_a: np.ndarray, uv_b: np.ndarray, cam_a: Camera, cam_b: Camera) -> np.ndarray:
    """Batched two-view DLT with the known projection matrices (metric, no alignment)."""
    a, b = np.asarray(uv_a, float).reshape(-1, 2), np.asarray(uv_b, float).reshape(-1, 2)
    if len(a) != len(b): raise ValueError("Match arrays must have the same length")
    if len(a) == 0: return np.empty((0, 3))
    P1, P2 = cam_a.P, cam_b.P
    A = np.stack((a[:, :1]*P1[2]-P1[0], a[:, 1:]*P1[2]-P1[1], b[:, :1]*P2[2]-P2[0], b[:, 1:]*P2[2]-P2[1]), 1)
    # Row normalization improves conditioning without changing the null space.
    A /= np.maximum(np.linalg.norm(A, axis=2, keepdims=True), 1e-12)
    x = np.linalg.svd(A)[2][:, -1]
    return x[:, :3] / x[:, 3:4]


def triangulation_angle_deg(xyz: np.ndarray, cam_a: Camera, cam_b: Camera) -> np.ndarray:
    va = xyz-cam_a.center; vb = xyz-cam_b.center
    cosine = np.sum(va*vb, 1) / np.maximum(np.linalg.norm(va, axis=1)*np.linalg.norm(vb, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def filter_pair(xyz: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray, cam_a: Camera, cam_b: Camera, reproj_px: float, min_angle_deg: float) -> np.ndarray:
    """Cheirality, max two-view reprojection error, and triangulation angle."""
    pa, za = cam_a.project(xyz); pb, zb = cam_b.project(xyz)
    errors = np.maximum(np.linalg.norm(pa-uv_a, axis=1), np.linalg.norm(pb-uv_b, axis=1))
    return (za > 0) & (zb > 0) & (errors < reproj_px) & (triangulation_angle_deg(xyz, cam_a, cam_b) >= min_angle_deg)
