"""Keyframe identity repair for propagated masks.

SAM2 propagates each camera independently, so in crowded plays it swaps identities in some views but
not others (mask-centroid rays of one "person" then miss each other by metres). At keyframes we
discard the IDs and rebuild them in 3D, exactly as Stage A does at frame 0: carve the court volume
with the union of all masks, take 3D components as people (view-consistent by construction), match
components to tracked people by distance to their last re-associated position plus jersey colour,
and relabel every view's segments from the projected components."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ring_init.masks.instances import find_instances, splat_silhouette
from ring_init.masks.segment import instance_histogram


class IdentityRepair:
    def __init__(self, s, cfg, person_ids: list[int]):
        self.s, self.cfg, self.ids = s, cfg, list(person_ids)
        info = s.instance_info(); labels0 = s.labels()
        self.ref_pos = {i["instance_id"]: np.asarray(i["center"]) for i in info["instances"] if i["instance_id"] in self.ids}
        self.ref_hist = {}
        for k in self.ref_pos:
            hs = [instance_histogram(s.images[c], labels0[c] == k) for c in range(len(labels0)) if (labels0[c] == k).sum() > 50]
            if hs: self.ref_hist[k] = np.median(np.stack(hs), 0).astype(np.float32)
        self.floor = s.floor()

    def __call__(self, labels: list[np.ndarray], images: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
        c, cams = self.cfg, self.s.cameras
        unions = [lab > 0 for lab in labels]; vox = c.m(c.occupancy_voxel_m)
        comps, stats = find_instances(cams, unions, self.floor, vox, c.m(c.occupancy_height_m), c.m(c.court_margin_m), c.occupancy_min_views,
                                      c.occupancy_allowed_misses, c.m(c.instance_min_height_m), c.instance_min_voxels, c.m(c.instance_split_footprint_m), c.device)
        dist_m = lambda a, b: float(np.linalg.norm(a - b)) / c.units_per_meter
        comps = self.seeded_split(comps)
        # component appearance: pixels of its projected silhouette that lie inside any mask
        sils = [[splat_silhouette(comp.voxels, cam, 2) for cam in cams] for comp in comps]
        hists = []
        for j, comp in enumerate(comps):
            hs = [instance_histogram(images[v], sils[j][v][0] & unions[v]) for v in range(len(cams)) if (sils[j][v][0] & unions[v]).sum() > 50]
            hists.append(np.median(np.stack(hs), 0).astype(np.float32) if hs else None)
        known = [k for k in self.ids if k in self.ref_pos]
        cost = np.full((len(known), len(comps)), 1e6)
        for a, k in enumerate(known):
            for b, comp in enumerate(comps):
                dm = dist_m(self.ref_pos[k], comp.center)
                if dm > c.reassoc_max_move_m: continue
                app = cv2.compareHist(self.ref_hist[k], hists[b], cv2.HISTCMP_BHATTACHARYYA) if (k in self.ref_hist and hists[b] is not None) else 0.5
                cost[a, b] = dm / c.reassoc_distance_scale_m + c.reassoc_appearance_weight * app
        rows, cols = linear_sum_assignment(cost) if len(comps) and len(known) else ((), ())
        match = {known[a]: b for a, b in zip(rows, cols) if cost[a, b] < 1e5}
        for k, b in match.items(): self.ref_pos[k] = comps[b].center
        # relabel each view: segments (any old id) -> the component whose silhouette explains them best
        new = []
        for v, (cam, lab) in enumerate(zip(cams, labels)):
            out = np.zeros_like(lab); segs = [int(x) for x in np.unique(lab) if x > 0]
            order = sorted(match.items(), key=lambda kv: -sils[kv[1]][v][1])        # far first; nearer overwrite
            for k, b in order:
                sil = sils[b][v][0]
                if not sil.any(): continue
                best, best_iou = None, c.reassoc_min_iou
                for sgid in segs:
                    m = lab == sgid; inter = (m & sil).sum()
                    if inter == 0: continue
                    iou = inter / (m | sil).sum()
                    if iou > best_iou: best, best_iou = sgid, iou
                region = (lab == best) & sil if best is not None else sil & (lab > 0)
                out[region] = k
            new.append(out)
        return new, {"components": len(comps), "matched": len(match), "unmatched_people": [k for k in known if k not in match]}

    def seeded_split(self, comps: list) -> list:
        """Players standing together carve into one component: split a component's voxels by the
        nearest (floor-plane) last-known position of the people whose position lies inside it."""
        from ring_init.masks.instances import Instance
        c = self.cfg; out = []
        refs = {k: self.floor.to_plane(p) for k, p in self.ref_pos.items()}
        for comp in comps:
            uv = self.floor.to_plane(comp.voxels); lo, hi = uv.min(0), uv.max(0); pad = c.m(c.reassoc_seed_margin_m)
            seeds = [k for k, r in refs.items() if np.all(r >= lo - pad) and np.all(r <= hi + pad)]
            if len(seeds) <= 1: out.append(comp); continue
            R = np.stack([refs[k] for k in seeds]); lab = np.argmin(((uv[:, None] - R[None]) ** 2).sum(-1), 1)
            for j in range(len(seeds)):
                sel = lab == j
                if sel.sum() >= c.instance_min_voxels: out.append(Instance(-1, comp.voxels[sel], uv[sel].mean(0), comp.height))
        return out
