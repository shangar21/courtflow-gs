"""Standard (INRIA 3DGS) PLY layout + non-optimized per-Gaussian instance ids sidecar."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from ring_init.gs.train import PARAM_KEYS, GaussianModel


def save_ply(model: GaussianModel, path: str | Path) -> None:
    """x y z nx ny nz f_dc_0..2 f_rest_* opacity(logit) scale_*(log) rot_*(wxyz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = {k: model.params[k].detach().float().cpu().numpy() for k in PARAM_KEYS}
    n = len(p["means"])
    f_dc = p["sh0"].reshape(n, 3)
    # INRIA stores features_rest as [N, 3, K-1] flattened (channel-major).
    f_rest = np.transpose(p["shN"], (0, 2, 1)).reshape(n, -1)
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + \
            [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] + \
            [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    values = np.concatenate([p["means"], np.zeros((n, 3), np.float32), f_dc, f_rest,
                             p["opacities"].reshape(n, 1), p["scales"], p["quats"]], 1).astype(np.float32)
    arr = np.empty(n, dtype=[(k, "f4") for k in names])
    for i, k in enumerate(names):
        arr[k] = values[:, i]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    np.save(path.with_name("instance_ids.npy"), model.params["instance_ids"].detach().cpu().numpy())


def load_ply(path: str | Path, device: str = "cuda") -> GaussianModel:
    path = Path(path)
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    col = lambda k: np.asarray(v[k], np.float32)
    n = len(v.data)
    rest = sorted([k for k in names if k.startswith("f_rest_")], key=lambda k: int(k.split("_")[-1]))
    f_rest = np.stack([col(k) for k in rest], 1) if rest else np.zeros((n, 0), np.float32)
    shN = np.transpose(f_rest.reshape(n, 3, -1), (0, 2, 1))
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    params = {
        "means": t(np.stack([col("x"), col("y"), col("z")], 1)),
        "sh0": t(np.stack([col(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]),
        "shN": t(shN),
        "opacities": t(col("opacity")),
        "scales": t(np.stack([col(f"scale_{i}") for i in range(3)], 1)),
        "quats": t(np.stack([col(f"rot_{i}") for i in range(4)], 1)),
    }
    ids_path = path.with_name("instance_ids.npy")
    ids = np.load(ids_path) if ids_path.is_file() else np.zeros(n, np.int32)
    return GaussianModel(params, torch.tensor(ids, dtype=torch.int32, device=device))


def concat_models(models: list[GaussianModel]) -> GaussianModel:
    """Compose instances (+ background) into one model for joint depth-sorted rendering."""
    if not models:
        raise ValueError("No models to concatenate")
    degrees = {m.sh_degree for m in models}
    if len(degrees) != 1:
        raise ValueError(f"Mixed SH degrees {degrees}")
    params = {k: torch.cat([m.params[k].detach() for m in models]) for k in PARAM_KEYS}
    ids = torch.cat([m.params["instance_ids"].detach() for m in models])
    return GaussianModel(params, ids)
