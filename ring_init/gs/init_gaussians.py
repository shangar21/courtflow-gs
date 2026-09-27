from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from ring_init.config import Config
from ring_init.gs.train import GaussianModel

SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """Quaternion wxyz rotating +z onto each normal."""
    n = np.asarray(normals, np.float32).copy()
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    z = np.array([0, 0, 1], np.float32)
    cross = np.cross(np.broadcast_to(z, n.shape), n)
    dot = (n * z).sum(1, keepdims=True)
    q = np.concatenate((1 + dot, cross), 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    opposite = dot[:, 0] < -0.9999
    q[opposite] = np.array([0, 1, 0, 0], np.float32)
    return q


def initialize_gaussians(points: np.ndarray, colors: np.ndarray, normals: np.ndarray, instance_id: int,
                         cfg: Config) -> GaussianModel:
    """Surfel-like Gaussians: tangent scale = mean distance to k nearest points, normal-axis scale
    = normal_scale_ratio x tangent, local z aligned with the normal, SH DC from colour (0..1)."""
    points = np.asarray(points, np.float64)
    k = cfg.knn_scale_k
    if len(points) < k + 1:
        raise ValueError(f"Need at least {k + 1} fused points to initialize Gaussians, got {len(points)}")
    distances, _ = cKDTree(points).query(points, k=k + 1)
    tangent = np.maximum(distances[:, 1:].mean(1), 1e-7)
    scale = np.stack((tangent, tangent, tangent * cfg.normal_scale_ratio), 1)
    device = cfg.device
    n = len(points)
    rgb = torch.tensor(np.clip(colors, 0, 1), dtype=torch.float32, device=device)
    sh_rest = (cfg.sh_degree + 1) ** 2 - 1
    params = {
        "means": torch.tensor(points, dtype=torch.float32, device=device),
        "scales": torch.log(torch.tensor(scale, dtype=torch.float32, device=device)),
        "quats": torch.tensor(normal_to_quaternion(normals), dtype=torch.float32, device=device),
        "opacities": torch.full((n,), float(np.log(cfg.opacity_init / (1 - cfg.opacity_init))), device=device),
        "sh0": rgb_to_sh(rgb)[:, None, :],
        "shN": torch.zeros((n, sh_rest, 3), dtype=torch.float32, device=device),
    }
    return GaussianModel(params, torch.full((n,), int(instance_id), dtype=torch.int32, device=device))
