"""Online per-frame tracking: only control-point states (t, q) of all persons are optimized; the
canonical Gaussians and the static background stay frozen.

Rendering: persons are rasterized with RGB (SH evaluated per camera, band 1 rotated by MLS) plus
one one-hot channel per instance, so the per-instance accumulated weight can be supervised with
that instance's propagated mask. The static background is rendered once per camera and
composited behind."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform import mls
from ring_init.deform.control_points import ControlTopology
from ring_init.deform.regularizers import arap, cp_rotations_from_mls, cp_rotations_from_quaternions, temporal_acceleration
from ring_init.gs.train import ssim_map

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


@dataclass
class Persons:
    """Concatenated frozen canonical person Gaussians."""
    means: torch.Tensor; quats: torch.Tensor; scales: torch.Tensor; opacities: torch.Tensor
    sh0: torch.Tensor; sh1: torch.Tensor | None; instance_ids: torch.Tensor; ids: list[int]
    onehot: torch.Tensor = field(init=False)

    def __post_init__(self):
        lookup = {k: i for i, k in enumerate(self.ids)}
        idx = torch.as_tensor([lookup[int(k)] for k in self.instance_ids.tolist()], device=self.means.device)
        self.onehot = F.one_hot(idx, len(self.ids)).float()

    @classmethod
    def from_models(cls, models: dict[int, object]) -> "Persons":
        ks = sorted(k for k in models if k != 0); p = [models[k].params for k in ks]
        cat = lambda key: torch.cat([x[key].detach() for x in p])
        sh1 = cat("shN") if p[0]["shN"].shape[1] >= 3 else None
        return cls(cat("means"), F.normalize(cat("quats"), dim=-1), torch.exp(cat("scales")), torch.sigmoid(cat("opacities")),
                   cat("sh0"), None if sh1 is None else sh1[:, :3], cat("instance_ids").long(), ks)


@dataclass
class CameraFrame:
    viewmat: torch.Tensor; K: torch.Tensor; width: int; height: int; center: torch.Tensor
    background: torch.Tensor            # [H,W,3] static background render
    image: torch.Tensor | None = None   # [H,W,3]
    labels: torch.Tensor | None = None  # [H,W] int
    box: tuple[int, int, int, int] | None = None
    valid_ids: torch.Tensor | None = None  # [I] bool: instance prompted/labelled in this camera


def eval_colors(sh0: torch.Tensor, sh1: torch.Tensor | None, means: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """gsplat's degree-0/1 SH evaluation (+0.5, clamp >= 0)."""
    c = SH_C0 * sh0[:, 0]
    if sh1 is not None:
        d = F.normalize(means - center, dim=-1); x, y, z = d.unbind(-1)
        c = c + SH_C1 * (-y[:, None] * sh1[:, 0] + z[:, None] * sh1[:, 1] - x[:, None] * sh1[:, 2])
    return (c + 0.5).clamp_min(0)


def render_persons(p: Persons, means: torch.Tensor, quats: torch.Tensor, sh1: torch.Tensor | None, cam: CameraFrame, box: tuple[int, int, int, int] | None = None, with_instances: bool = False):
    """Returns rgb [h,w,3], per-instance weight [h,w,I] (None unless with_instances), alpha [h,w] for the crop `box`."""
    x0, y0, x1, y1 = box or (0, 0, cam.width, cam.height)
    K = cam.K.clone(); K[0, 2] -= x0; K[1, 2] -= y0
    feats = eval_colors(p.sh0, sh1, means, cam.center)
    if with_instances: feats = torch.cat((feats, p.onehot), 1)
    out, alpha, _ = rasterization(means, quats, p.scales, p.opacities, feats, cam.viewmat[None], K[None], x1 - x0, y1 - y0, sh_degree=None, packed=False)
    return out[0, ..., :3], (out[0, ..., 3:] if with_instances else None), alpha[0, ..., 0]


@dataclass
class FrameState:
    t: torch.Tensor
    q: torch.Tensor


def warm_start(prev: FrameState, prev2: FrameState | None, enabled: bool) -> FrameState:
    if not enabled or prev2 is None: return FrameState(prev.t.clone(), prev.q.clone())
    q2 = torch.where(((prev.q * prev2.q).sum(-1, keepdim=True) < 0), -prev2.q, prev2.q)
    return FrameState(2 * prev.t - prev2.t, F.normalize(2 * prev.q - q2, dim=-1))


class Tracker:
    def __init__(self, persons: Persons, topo: ControlTopology, cfg: Config):
        self.p, self.topo, self.cfg = persons, topo, cfg
        self.spacing = topo.spacing
        self.cam_cycle: list[int] = []
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.ids_t = torch.as_tensor(persons.ids, device=persons.means.device)
        lookup = {k: i for i, k in enumerate(persons.ids)}
        self.cp_inst_index = torch.as_tensor([lookup[int(k)] for k in topo.cp_instance.tolist()], device=topo.rest.device)

    def deform(self, t: torch.Tensor, q: torch.Tensor):
        c = self.cfg
        return mls.deform(self.p.means, self.p.quats, self.p.sh1, self.topo.rest, self.topo.gaussian_neighbors, self.topo.gaussian_weights,
                          t, q, c.mls_blend_cp_rotation, c.mls_rotation_blend_weight, c.mls_svd_epsilon)

    def next_cameras(self, n_cams: int) -> list[int]:
        k = self.cfg.cameras_per_iteration
        if k >= n_cams: return list(range(n_cams))
        while len(self.cam_cycle) < k:  # rotate through every camera before repeating
            self.cam_cycle += torch.randperm(n_cams, generator=self.gen).tolist()
        chosen, self.cam_cycle = self.cam_cycle[:k], self.cam_cycle[k:]
        return chosen

    def photometric(self, t, q, cams: list[CameraFrame], idx: list[int], timers: dict) -> torch.Tensor:
        c = self.cfg; sync = c.tracking_profile
        def mark(key, t0):
            if sync: torch.cuda.synchronize()
            timers[key] += time.perf_counter() - t0; return time.perf_counter()
        t0 = time.perf_counter()
        means, quats, sh1 = self.deform(t, q); t0 = mark("mls_fwd", t0)
        loss = 0.0
        for i in idx:
            cam = cams[i]; x0, y0, x1, y1 = cam.box
            rgb, inst, alpha = render_persons(self.p, means, quats, sh1, cam, cam.box, with_instances=c.tracking_mask_mode == "instance")
            bg = cam.background[y0:y1, x0:x1]; img = cam.image[y0:y1, x0:x1]; lab = cam.labels[y0:y1, x0:x1]
            comp = rgb + (1 - alpha[..., None]) * bg
            region = F.max_pool2d((lab > 0).float()[None, None], 2 * c.tracking_loss_margin_px + 1, 1, c.tracking_loss_margin_px)[0, 0]
            m = region[..., None]; denom = region.sum().clamp_min(1)
            l1 = ((comp - img).abs() * m).sum() / (3 * denom)
            pr, tg = (comp * m).permute(2, 0, 1), (img * m).permute(2, 0, 1)
            dssim = ((1 - ssim_map(pr, tg, c.ssim_window)) * region).sum() / (3 * denom)
            loss = loss + c.l1_weight * l1 + c.dssim_weight * dssim
            if c.tracking_alpha_weight > 0:
                if inst is not None:
                    target = (lab[..., None] == self.ids_t).float()
                    weight = cam.valid_ids.float()[None, None, :].expand_as(target)   # weights, not boolean indexing (no sync)
                    bce = F.binary_cross_entropy(inst.clamp(1e-5, 1 - 1e-5), target, weight=weight, reduction="sum") / weight.sum().clamp_min(1)
                else:
                    bce = F.binary_cross_entropy(alpha.clamp(1e-5, 1 - 1e-5), (lab > 0).float())
                loss = loss + c.tracking_alpha_weight * bce
        mark("render", t0)
        return loss / max(len(idx), 1)

    def track_frame(self, cams: list[CameraFrame], init: FrameState, prev: FrameState | None, prev2: FrameState | None,
                    anchor: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[FrameState, dict]:
        """anchor = (target per-person mean translation [I,3], valid [I]) from triangulated mask centroids."""
        c = self.cfg; sync = c.tracking_profile; inst = self.cp_inst_index
        # t_j = T_person(j) + delta_j: the shared per-person translation receives the summed gradient
        # of the whole body (fast bulk motion); delta_j carries articulation. Adam's per-coordinate
        # normalization moves every control ~lr per step regardless of its gradient, so a large eps
        # (tracking_adam_eps) keeps weakly observed controls from random-walking.
        if c.tracking_global_translation:
            T0 = torch.zeros(len(self.p.ids), 3, device=init.t.device).index_reduce_(0, inst, init.t, "mean", include_self=False)
            T = torch.nn.Parameter(T0); delta = torch.nn.Parameter(init.t - T0[inst])
            translation = lambda: T[inst] + delta
            groups = [{"params": [T], "lr": c.m(c.tracking_lr_global_m)}, {"params": [delta], "lr": c.m(c.tracking_lr_translation_m)}]
        else:
            delta = torch.nn.Parameter(init.t.clone()); translation = lambda: delta
            groups = [{"params": [delta], "lr": c.m(c.tracking_lr_translation_m)}]
        q = torch.nn.Parameter(init.q.clone())
        opt = torch.optim.Adam(groups + [{"params": [q], "lr": c.tracking_lr_rotation}], eps=c.tracking_adam_eps)
        timers = {k: 0.0 for k in ("mls_fwd", "render", "backward", "optimizer", "regularizers")}; history = []; it = 0; n = c.early_stop_patience
        for it in range(c.tracking_iterations):
            idx = self.next_cameras(len(cams)); tt = translation()
            loss = self.photometric(tt, F.normalize(q, dim=-1), cams, idx, timers)
            a = time.perf_counter()
            if c.arap_weight > 0:
                R = cp_rotations_from_quaternions(F.normalize(q, dim=-1)) if c.tracking_optimize_rotations else cp_rotations_from_mls(self.topo.rest, tt, self.topo.graph_neighbors)
                loss = loss + c.arap_weight * arap(self.topo.rest, tt, self.topo.graph_neighbors, R)
            if c.temporal_weight > 0 and prev is not None and prev2 is not None:
                loss = loss + c.temporal_weight * temporal_acceleration(tt, prev.t, prev2.t, self.spacing)
            if anchor is not None and c.tracking_anchor_weight > 0 and bool(anchor[1].any()):
                mean_t = torch.zeros_like(anchor[0]).index_reduce_(0, inst, tt, "mean", include_self=False)
                r = (mean_t - anchor[0]).norm(dim=-1) / c.m(c.tracking_anchor_scale_m)
                loss = loss + c.tracking_anchor_weight * F.huber_loss(r[anchor[1]], torch.zeros_like(r[anchor[1]]), delta=1.0)
            if sync: torch.cuda.synchronize()
            timers["regularizers"] += time.perf_counter() - a; a = time.perf_counter()
            opt.zero_grad(set_to_none=True); loss.backward()
            if sync: torch.cuda.synchronize()
            timers["backward"] += time.perf_counter() - a; a = time.perf_counter()
            opt.step()
            if sync: torch.cuda.synchronize()
            timers["optimizer"] += time.perf_counter() - a
            history.append(loss.detach())
            # Single-iteration losses use different camera subsets; compare window means.
            if len(history) >= 2 * n and it >= c.tracking_min_iterations and it % 5 == 0:
                h = torch.stack(history[-2 * n:]); old, new = float(h[:n].mean()), float(h[n:].mean())
                if old - new < c.early_stop_delta * max(abs(old), 1e-12): break
        state = FrameState(translation().detach(), F.normalize(q.detach(), dim=-1))
        w = min(len(history), max(1, n))
        return state, {**timers, "iterations": it + 1, "final_loss": float(torch.stack(history[-w:]).mean()), "first_loss": float(torch.stack(history[:w]).mean())}


def identity_state(m: int, device) -> FrameState:
    q = torch.zeros(m, 4, device=device); q[:, 0] = 1
    return FrameState(torch.zeros(m, 3, device=device), q)


def mask_centroid_anchor(cams: list[CameraFrame], ids: list[int], min_px: int, min_views: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-instance 3D point closest (least squares, area-weighted) to the rays through the
    instance's mask centroid in every camera. Returns (points [I,3], valid [I])."""
    dev = cams[0].viewmat.device; I = len(ids)
    A = torch.zeros(I, 3, 3, device=dev); b = torch.zeros(I, 3, device=dev); n = torch.zeros(I, device=dev)
    ids_t = torch.as_tensor(ids, device=dev)
    for cam in cams:
        if cam.labels is None: continue
        onehot = (cam.labels[..., None] == ids_t).float()                       # [H,W,I]
        area = onehot.sum((0, 1))
        ys = torch.arange(cam.height, device=dev, dtype=torch.float32); xs = torch.arange(cam.width, device=dev, dtype=torch.float32)
        cy = (onehot.sum(1) * ys[:, None]).sum(0) / area.clamp_min(1); cx = (onehot.sum(0) * xs[:, None]).sum(0) / area.clamp_min(1)
        R = cam.viewmat[:3, :3]; C = -R.T @ cam.viewmat[:3, 3]
        d = (R.T @ (torch.linalg.inv(cam.K) @ torch.stack((cx, cy, torch.ones_like(cx)))))
        d = (d / d.norm(dim=0, keepdim=True)).T                                   # [I,3]
        P = torch.eye(3, device=dev) - d[:, :, None] * d[:, None, :]
        w = (area >= min_px).float() * area.sqrt()
        A += w[:, None, None] * P; b += w[:, None] * (P @ C); n += (area >= min_px).float()
    valid = n >= min_views
    X = torch.linalg.solve(A + 1e-9 * torch.eye(3, device=dev), b[..., None])[..., 0]
    return X, valid
