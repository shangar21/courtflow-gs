"""GPU voxel carving: court-wide person occupancy and strict per-instance visual hulls."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from ring_init.io.calib import Camera


def project_torch(xyz: torch.Tensor, cam: Camera) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns integer pixel (u, v), depth z, and in-frustum flag for (N,3) points."""
    R = torch.as_tensor(cam.R, dtype=xyz.dtype, device=xyz.device); t = torch.as_tensor(cam.t, dtype=xyz.dtype, device=xyz.device)
    K = torch.as_tensor(cam.K, dtype=xyz.dtype, device=xyz.device)
    pc = xyz @ R.T + t; z = pc[:, 2]
    uvw = pc @ K.T; uv = uvw[:, :2] / uvw[:, 2:].clamp_min(1e-9)
    u = torch.floor(uv[:, 0] + 0.5).long(); v = torch.floor(uv[:, 1] + 0.5).long()
    inside = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return u.clamp(0, cam.width-1), v.clamp(0, cam.height-1), z, inside


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0: return mask
    import cv2
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))) > 0


def occupancy(points: torch.Tensor, cameras: list[Camera], masks: list[np.ndarray], min_views: int, allowed_misses: int, chunk: int = 4_000_000) -> torch.Tensor:
    """Voxel centres inside the mask in >= max(min_views, visible - allowed_misses) views."""
    keep = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    mask_t = [torch.as_tensor(m, device=points.device) for m in masks]
    for s in range(0, len(points), chunk):
        p = points[s:s+chunk]; visible = torch.zeros(len(p), dtype=torch.int32, device=p.device); hits = torch.zeros_like(visible)
        for cam, m in zip(cameras, mask_t):
            u, v, _, inside = project_torch(p, cam)
            visible += inside.int(); hits += (inside & m[v, u]).int()
        keep[s:s+chunk] = (hits >= torch.clamp(visible - allowed_misses, min=min_views)) & (visible >= min_views)
    return keep


def instance_hull(lo: np.ndarray, hi: np.ndarray, voxel: float, cameras: list[Camera], labels: list[np.ndarray], instance_id: int, instance_depths: list[dict[int, float]], occluder_depths: list[np.ndarray] | None = None, occluder_margin: float = 0.0, allowed_misses: int = 0, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Occlusion-aware hull. A camera "misses" a voxel when the voxel is in frustum and its pixel
    carries neither this instance's label, nor a nearer instance (dynamic occluder), nor static
    geometry nearer than the voxel by `occluder_margin` (occluder_depths: per-camera min depth of
    known-pose background points, inf where unknown). Voxels with <= allowed_misses survive.
    Returns (occupied voxel centres, surface flag)."""
    axes = [torch.arange(lo[d], hi[d] + voxel*0.5, voxel, device=device, dtype=torch.float64) for d in range(3)]
    shape = tuple(len(a) for a in axes)
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    misses = torch.zeros(len(grid), dtype=torch.int32, device=device); seen = torch.zeros(len(grid), dtype=torch.int32, device=device)
    for ci, (cam, lab) in enumerate(zip(cameras, labels)):
        if instance_id not in instance_depths[ci]: continue  # no mask in this view: it cannot carve
        lab_t = torch.as_tensor(lab, device=device)
        u, v, z, inside = project_torch(grid, cam)
        pix = lab_t[v, u]
        own = pix == instance_id
        my_depth = instance_depths[ci].get(instance_id, np.inf)
        occluders = [k for k, d in instance_depths[ci].items() if k != instance_id and d < my_depth]
        occluded = torch.isin(pix, torch.as_tensor(occluders, device=device, dtype=pix.dtype)) if occluders else torch.zeros_like(own)
        if occluder_depths is not None:
            occ_d = torch.as_tensor(occluder_depths[ci], device=device, dtype=z.dtype)
            occluded = occluded | (occ_d[v, u] < z - occluder_margin)
        misses += (inside & ~own & ~occluded).int()
        seen += (inside & own).int()
    keep = (misses <= allowed_misses) & (seen >= 2)
    occ = keep.reshape(shape)
    # Surface: occupied voxels with at least one empty 6-neighbour.
    o = occ.float()[None, None]
    kernel = torch.zeros((1, 1, 3, 3, 3), device=device); kernel[0, 0, 1, 1, :] = 1; kernel[0, 0, 1, :, 1] = 1; kernel[0, 0, :, 1, 1] = 1
    neighbours = F.conv3d(F.pad(o, (1, 1, 1, 1, 1, 1)), kernel)[0, 0]
    surface = occ & (neighbours < 7)
    centres = grid[occ.reshape(-1)].cpu().numpy()
    return centres, surface.reshape(-1)[occ.reshape(-1)].cpu().numpy()


def inside_hull(points: np.ndarray, occupied: np.ndarray, voxel: float, dilation_voxels: int) -> np.ndarray:
    """Membership test of points against a set of occupied voxel centres (with dilation)."""
    if len(occupied) == 0 or len(points) == 0: return np.zeros(len(points), bool)
    origin = occupied.min(0) - (dilation_voxels + 1) * voxel
    idx = np.floor((occupied - origin) / voxel + 0.5).astype(np.int64)
    shape = idx.max(0) + dilation_voxels + 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    occ = torch.zeros(tuple(shape.tolist()), dtype=torch.float32, device=device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    if dilation_voxels > 0:
        # Separable box dilation (equal to a cubic max-pool, far cheaper for large radii).
        k, r = 2*dilation_voxels + 1, dilation_voxels
        o = occ[None, None]
        for kernel, pad in (((k, 1, 1), (r, 0, 0)), ((1, k, 1), (0, r, 0)), ((1, 1, k), (0, 0, r))):
            o = F.max_pool3d(o, kernel, stride=1, padding=pad)
        occ = o[0, 0]
    occ = occ.cpu()
    q = np.floor((np.asarray(points) - origin) / voxel + 0.5).astype(np.int64)
    ok = np.all((q >= 0) & (q < shape), 1)
    result = np.zeros(len(points), bool)
    result[ok] = occ[q[ok, 0], q[ok, 1], q[ok, 2]].numpy() > 0
    return result


def occluder_depth_maps(points: np.ndarray, cameras: list[Camera], splat_px: int) -> list[np.ndarray]:
    """Per camera: min depth of known-pose points splatted with a (2r+1)^2 min filter; inf = none."""
    import cv2
    maps = []
    for cam in cameras:
        uv, z = cam.project(points); u = np.rint(uv[:, 0]).astype(int); v = np.rint(uv[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        buf = np.full((cam.height, cam.width), np.inf, np.float32); np.minimum.at(buf, (v[ok], u[ok]), z[ok].astype(np.float32))
        maps.append(cv2.erode(buf, np.ones((2*splat_px+1, 2*splat_px+1), np.uint8)) if splat_px > 0 else buf)
    return maps


class HullMembership:
    """GPU membership test against a voxel hull (dilated), for pruning Gaussian centres."""

    def __init__(self, occupied: np.ndarray, voxel: float, dilation_voxels: int, device: str = "cuda"):
        self.voxel = voxel; pad = dilation_voxels + 1
        self.origin = torch.as_tensor(occupied.min(0) - pad * voxel, dtype=torch.float32, device=device)
        idx = torch.as_tensor(np.floor((occupied - self.origin.cpu().numpy()) / voxel + 0.5).astype(np.int64), device=device)
        shape = (idx.max(0).values + pad + 1).tolist()
        occ = torch.zeros(shape, dtype=torch.float32, device=device); occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        if dilation_voxels > 0:
            k, r = 2*dilation_voxels + 1, dilation_voxels; o = occ[None, None]
            for kernel, padding in (((k, 1, 1), (r, 0, 0)), ((1, k, 1), (0, r, 0)), ((1, 1, k), (0, 0, r))):
                o = F.max_pool3d(o, kernel, stride=1, padding=padding)
            occ = o[0, 0]
        self.occ = occ > 0; self.shape = torch.as_tensor(shape, device=device)

    def inside(self, points: torch.Tensor) -> torch.Tensor:
        q = torch.floor((points.detach().float() - self.origin) / self.voxel + 0.5).long()
        ok = ((q >= 0) & (q < self.shape)).all(1)
        result = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        qq = q[ok]; result[ok] = self.occ[qq[:, 0], qq[:, 1], qq[:, 2]]
        return result

    def outside(self, points: torch.Tensor) -> torch.Tensor:
        return ~self.inside(points)
