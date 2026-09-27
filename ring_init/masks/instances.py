"""Cross-view instance identity from 3D person occupancy (no 2D centroid chaining).

Every camera's union person mask carves a voxel grid over the court volume; 3D connected
components are people. Each component is projected back into every camera to (a) pick the SAM2
detection mask it explains, or (b) re-prompt SAM2 where the detector missed it. Overlapping
pixels go to the instance nearest to that camera, so labels are mutually exclusive."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from ring_init.geom.floor import Floor
from ring_init.geom.hull import dilate, occupancy, project_torch
from ring_init.io.calib import Camera


@dataclass
class Instance:
    instance_id: int
    voxels: np.ndarray        # occupancy voxel centres (world)
    floor_uv: np.ndarray      # footprint centre in floor coordinates
    height: float             # scene units

    @property
    def center(self) -> np.ndarray:
        return self.voxels.mean(0)


def court_grid(floor: Floor, voxel: float, height: float, margin: float, device: str) -> tuple[torch.Tensor, tuple[int, int, int], np.ndarray]:
    lo = floor.extent_lo - margin; hi = floor.extent_hi + margin
    us = np.arange(lo[0], hi[0], voxel); vs = np.arange(lo[1], hi[1], voxel); hs = np.arange(voxel/2, height, voxel)
    U, V, H = np.meshgrid(us, vs, hs, indexing="ij")
    world = floor.from_plane(np.stack((U, V), -1).reshape(-1, 2), H.reshape(-1))
    return torch.as_tensor(world, device=device), U.shape, np.stack((U, V, H), -1).reshape(-1, 3)


def find_instances(cameras: list[Camera], unions: list[np.ndarray], floor: Floor, voxel: float, height: float, margin: float, min_views: int, allowed_misses: int, min_height: float, min_voxels: int, split_footprint: float, device: str = "cuda") -> tuple[list[Instance], dict]:
    grid, shape, fuv = court_grid(floor, voxel, height, margin, device)
    keep = occupancy(grid, cameras, unions, min_views, allowed_misses).reshape(shape).cpu().numpy()
    comp, count = ndimage.label(keep, structure=np.ones((3, 3, 3)))
    fuv = fuv.reshape(*shape, 3); world = grid.cpu().numpy().reshape(*shape, 3)
    raw, rejected = [], []
    for c in range(1, count+1):
        sel = comp == c; n = int(sel.sum()); pts = world[sel]; f = fuv[sel]
        h_extent = f[:, 2].max() - f[:, 2].min() + voxel; bottom = f[:, 2].min()
        if n < min_voxels or h_extent < min_height or bottom > 2*voxel + 0.25*min_height:
            rejected.append({"voxels": n, "height": float(h_extent), "bottom": float(bottom)}); continue
        extent = f[:, :2].max(0) - f[:, :2].min(0)
        parts = max(1, int(np.ceil(float(extent.max()) / split_footprint)))
        if parts == 1:
            raw.append((pts, f)); continue
        from scipy.cluster.vq import kmeans2
        _, lab = kmeans2(f[:, :2], parts, seed=0, minit="++")
        for k in range(parts):
            if (lab == k).sum() >= min_voxels: raw.append((pts[lab == k], f[lab == k]))
    # Deterministic IDs: sort by floor position (u, then v).
    raw.sort(key=lambda x: (round(float(x[1][:, 0].mean()), 3), float(x[1][:, 1].mean())))
    instances = [Instance(i+1, p, f[:, :2].mean(0), float(f[:, 2].max()-f[:, 2].min()+voxel)) for i, (p, f) in enumerate(raw)]
    stats = {"occupied_voxels": int(keep.sum()), "components": int(count), "rejected": rejected, "instances": len(instances)}
    return instances, stats


def splat_silhouette(points: np.ndarray, cam: Camera, radius_px: int, device: str = "cuda") -> tuple[np.ndarray, float]:
    """Binary silhouette of projected voxel centres (dilated) and median depth."""
    p = torch.as_tensor(points, device=device)
    u, v, z, inside = project_torch(p, cam)
    sil = torch.zeros((cam.height, cam.width), dtype=torch.bool, device=device)
    sil[v[inside], u[inside]] = True
    sil = dilate(sil.cpu().numpy(), radius_px)
    return sil, float(z[inside].median()) if bool(inside.any()) else float("inf")


def label_cameras(instances: list[Instance], cameras: list[Camera], person: list[dict], refiner, images: list[np.ndarray], voxel_radius_px: int, min_iou: float, min_area_px: int, max_hist_distance: float) -> tuple[list[np.ndarray], list[dict[int, float]], list[dict]]:
    """Per camera int32 label image (0 = background), per-instance depth, and a report.

    Pass 1 assigns detector masks to projected occupancy silhouettes (Hungarian on IoU). Pass 2
    re-prompts SAM2 where the detector missed an instance; the re-prompted mask is accepted only if
    its colour histogram matches the instance's detector masks in other views (otherwise the
    person is occluded by static structure there and the view carries no label for it)."""
    import cv2
    from ring_init.masks.segment import instance_histogram
    per_cam = []
    for ci, cam in enumerate(cameras):
        sils, zs = zip(*[splat_silhouette(inst.voxels, cam, voxel_radius_px) for inst in instances])
        det = person[ci]["masks"]
        cost = np.ones((len(instances), len(det)))
        for i, sil in enumerate(sils):
            if not sil.any(): continue
            for j, m in enumerate(det):
                inter = np.logical_and(sil, m).sum()
                if inter: cost[i, j] = 1 - inter / np.logical_or(sil, m).sum()
        rows, cols = linear_sum_assignment(cost) if len(det) else ((), ())
        chosen = {i: det[j] & sils[i] for i, j in zip(rows, cols) if 1 - cost[i, j] >= min_iou}
        per_cam.append((sils, zs, chosen))
    reference = {}
    for i in range(len(instances)):
        hists = [instance_histogram(images[ci], chosen[i]) for ci, (_, _, chosen) in enumerate(per_cam) if i in chosen and chosen[i].sum() >= min_area_px]
        if hists: reference[i] = np.median(np.stack(hists), 0).astype(np.float32)
    labels, depths, report = [], [], []
    for ci, cam in enumerate(cameras):
        sils, zs, chosen = per_cam[ci]; matched = len(chosen)
        missing = [i for i, sil in enumerate(sils) if i not in chosen and sil.sum() >= min_area_px]
        reprompted, rejected = [], []
        if missing:
            refiner.set_image(images[ci])
            boxes, points = [], []
            for i in missing:
                ys, xs = np.nonzero(sils[i]); boxes.append([xs.min(), ys.min(), xs.max(), ys.max()]); points.append([np.median(xs), np.median(ys)])
            masks, _ = refiner.masks(np.asarray(boxes, float), np.asarray(points, float))
            for i, m in zip(missing, masks):
                m = m & sils[i]
                dist = cv2.compareHist(instance_histogram(images[ci], m).astype(np.float32), reference[i], cv2.HISTCMP_BHATTACHARYYA) if i in reference and m.sum() else 1.0
                if dist <= max_hist_distance: chosen[i] = m; reprompted.append((instances[i].instance_id, round(float(dist), 3)))
                else: rejected.append((instances[i].instance_id, round(float(dist), 3)))
        # Mutually exclusive labels: far first, nearer instances overwrite shared pixels.
        lab = np.zeros((cam.height, cam.width), np.int32)
        for i in sorted(chosen, key=lambda k: -zs[k]):
            if chosen[i].sum() >= min_area_px: lab[chosen[i]] = instances[i].instance_id
        depth_of = {instances[i].instance_id: zs[i] for i in chosen if (lab == instances[i].instance_id).sum() >= min_area_px}
        labels.append(lab); depths.append(depth_of)
        report.append({"camera": ci, "detections": len(person[ci]["masks"]), "matched": matched, "reprompted": reprompted, "rejected": rejected, "labelled": sorted(depth_of)})
    return labels, depths, report
