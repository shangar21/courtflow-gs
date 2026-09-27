"""Known-pose Gaussian optimization for one instance (crop views) or the static background.

Camera poses and intrinsics are inputs only; they are never parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable
import json
import math
import time

import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, remove, split

from ring_init.config import Config

PARAM_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


class GaussianModel:
    """Gaussian parameters in gsplat's convention (log scales, logit opacities, wxyz quats).

    `instance_ids` lives in the same ParameterDict as a non-trainable entry so that gsplat's
    duplicate/split/remove ops keep it aligned with every other per-Gaussian tensor.
    """

    def __init__(self, params: dict[str, torch.Tensor], instance_ids: torch.Tensor):
        self.params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(params[k].detach().contiguous().float(), requires_grad=True) for k in PARAM_KEYS})
        self.params["instance_ids"] = torch.nn.Parameter(instance_ids.detach().to(torch.int32).contiguous(), requires_grad=False)

    def __len__(self) -> int:
        return len(self.params["means"])

    def __getattr__(self, name: str):
        if name != "params" and "params" in self.__dict__ and name in self.__dict__["params"]:
            return self.__dict__["params"][name]
        raise AttributeError(name)

    @property
    def sh_degree(self) -> int:
        return int(math.isqrt(self.params["shN"].shape[1] + 1)) - 1

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    def detached(self) -> "GaussianModel":
        return GaussianModel({k: self.params[k].detach().clone() for k in PARAM_KEYS}, self.params["instance_ids"].detach().clone())


@dataclass
class View:
    """One calibrated training view. `loss_mask` selects photometric pixels, `alpha_target` is the
    instance mask, `alpha_ignore` marks pixels whose alpha is unknown (other instances in front)."""
    image: torch.Tensor                    # [3,H,W] float in [0,1]
    loss_mask: torch.Tensor                # [H,W] bool
    alpha_target: torch.Tensor             # [H,W] float
    alpha_ignore: torch.Tensor             # [H,W] bool
    viewmat: torch.Tensor                  # [4,4] world-to-camera
    K: torch.Tensor                        # [3,3]
    width: int
    height: int
    sparse_depth: torch.Tensor | None = None  # [H,W], 0 = no sample
    name: str = ""
    full_width: int | None = None          # size of the uncropped image (densification normalization)
    full_height: int | None = None


TILE = 16  # gsplat tile size


def crop_view(view: View, box: tuple[int, int, int, int], align: int = TILE) -> View:
    """Crop to integer box (x0,y0,x1,y1), exclusive max; shifts the principal point.

    The box is snapped outward to gsplat's 16-px tile grid so a crop render is
    pixel-identical to the same region of the full render (unaligned crops change per-tile
    culling and differ by up to ~1e-2)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if align > 1:
        x0, y0 = (x0 // align) * align, (y0 // align) * align
        x1, y1 = -(-x1 // align) * align, -(-y1 // align) * align
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, view.width), min(y1, view.height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty crop {box} for view {view.name}")
    K = view.K.clone()
    K[0, 2] -= x0
    K[1, 2] -= y0
    sl = (slice(y0, y1), slice(x0, x1))
    return replace(
        view, image=view.image[:, sl[0], sl[1]], loss_mask=view.loss_mask[sl], alpha_target=view.alpha_target[sl],
        alpha_ignore=view.alpha_ignore[sl], K=K, width=x1 - x0, height=y1 - y0,
        sparse_depth=None if view.sparse_depth is None else view.sparse_depth[sl],
        full_width=view.full_width or view.width, full_height=view.full_height or view.height)


def drop_multiplier(n: int, drop_rate: float, device) -> torch.Tensor | None:
    """DropGaussian (Park et al., CVPR 2025): each Gaussian is removed with probability r and the
    survivors' opacity is scaled by 1/(1-r). Training only; None means no dropping."""
    if drop_rate <= 0:
        return None
    return (torch.rand(n, device=device) >= drop_rate).float() / (1.0 - drop_rate)


def _activated(model: GaussianModel) -> tuple[torch.Tensor, ...]:
    p = model.params
    return (p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            torch.cat([p["sh0"], p["shN"]], 1))


def render_batch(model: GaussianModel, viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int,
                 drop_rate: float = 0.0, bg: torch.Tensor | None = None, drop_mult: torch.Tensor | None = None,
                 frozen: GaussianModel | None = None, radius_clip: float = 0.0):
    """Render C cameras of one raster size in a single gsplat call.

    Returns rgb [C,H,W,3], expected depth [C,H,W], alpha [C,H,W], gsplat info. `drop_mult`
    (from `drop_multiplier`) overrides `drop_rate` so one DropGaussian sample can be shared.
    `frozen` Gaussians are concatenated AFTER the trained ones (detached, never dropped) so
    compositing is depth-ordered; info["n_trained"] marks the trained slice of per-Gaussian
    outputs (means2d, radii)."""
    means, quats, scales, opacities, colors = _activated(model)
    n_trained = len(means)
    if drop_mult is None:
        drop_mult = drop_multiplier(n_trained, drop_rate, opacities.device)
    if drop_mult is not None:
        opacities = opacities * drop_mult
    sh_degree = model.sh_degree
    if frozen is not None and len(frozen) > 0:
        if frozen.sh_degree != sh_degree:
            raise ValueError(f"Frozen SH degree {frozen.sh_degree} != trained {sh_degree}")
        with torch.no_grad():
            fz = _activated(frozen)
        means, quats, scales, opacities, colors = (torch.cat([t, f.detach()]) for t, f in
                                                   zip((means, quats, scales, opacities, colors), fz))
    C = viewmats.shape[0]
    out, alpha, info = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree=sh_degree,
        packed=False, render_mode="RGB+ED", backgrounds=None if bg is None else bg.expand(C, -1), radius_clip=radius_clip)
    info["n_trained"] = n_trained
    return out[..., :3], out[..., 3], alpha[..., 0], info


def render(model: GaussianModel, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int,
           drop_rate: float = 0.0, bg: torch.Tensor | None = None, frozen: GaussianModel | None = None):
    """Single-camera render: rgb [H,W,3], expected depth [H,W], alpha [H,W], gsplat info.
    drop_rate > 0 applies DropGaussian (training only; 0 at eval)."""
    rgb, depth, alpha, info = render_batch(model, viewmat[None], K[None], width, height, drop_rate,
                                           None if bg is None else bg[None], frozen=frozen)
    return rgb[0], depth[0], alpha[0], info


_WINDOWS: dict = {}


def _gaussian_window(size: int, sigma: float, device, channels: int) -> torch.Tensor:
    """Cached contiguous depthwise 2-D Gaussian window (strided/expanded weights hit slow convs;
    a separable 1-D pair was measured slower in the backward pass)."""
    key = (size, sigma, str(device), channels)
    if key not in _WINDOWS:
        x = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        _WINDOWS[key] = (g[:, None] * g[None, :])[None, None].repeat(channels, 1, 1, 1).contiguous()
    return _WINDOWS[key]


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Windowed Gaussian SSIM (Wang et al. 2004, sigma 1.5) of [3,H,W] or [C,3,H,W] images,
    returning a map of the same shape. Uses the fused CUDA kernel (ring_init.gs.fused_ssim) when
    possible (11x11 window, CUDA float32, target without gradient); otherwise the PyTorch path."""
    if window == 11 and x.is_cuda and x.dtype == torch.float32 and not y.requires_grad:
        from ring_init.gs import fused_ssim
        if fused_ssim.available():
            return fused_ssim.fused_ssim_map(x, y)
    return ssim_map_torch(x, y, window)


def dssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, window: int = 11) -> torch.Tensor:
    """Mean D-SSIM; with a mask, the same masked form the trainer uses:
    mean((1 - ssim(pred*m, target*m)) * m)."""
    if mask is None:
        return (1 - ssim_map(pred, target, window)).mean()
    m = mask.float()
    while m.dim() < pred.dim(): m = m.unsqueeze(-3)
    return ((1 - ssim_map(pred * m, target * m, window)) * m).mean()


def ssim_map_torch(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Reference PyTorch implementation of `ssim_map`."""
    single = x.dim() == 3
    if single:
        x, y = x[None], y[None]
    ch, pad = x.shape[1], window // 2
    w = _gaussian_window(window, 1.5, x.device, ch)
    blur = lambda z: F.conv2d(z, w, padding=pad, groups=ch)
    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x ** 2
    syy = blur(y * y) - mu_y ** 2
    sxy = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s[0] if single else s


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    return ssim_map(x, y, window).mean()


def _project(means: torch.Tensor, view: View) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam = means @ view.viewmat[:3, :3].T + view.viewmat[:3, 3]
    z = cam[:, 2]
    uv = cam @ view.K.T
    uv = uv[:, :2] / z.clamp_min(1e-8)[:, None]
    return uv[:, 0], uv[:, 1], z


@torch.no_grad()
def mask_outside_counts(model: GaussianModel, views: list[View], dilate_px: int = 3) -> torch.Tensor:
    """Per Gaussian: number of views where its centre is in the image, not in an ignored
    (occluded) pixel, and outside the dilated instance mask."""
    counts = torch.zeros(len(model), device=model.device, dtype=torch.int32)
    for view in views:
        target = view.alpha_target[None, None].float()
        if dilate_px > 0:
            target = F.max_pool2d(target, 2 * dilate_px + 1, 1, dilate_px)
        target = target[0, 0] > 0.5
        u, v, z = _project(model.params["means"].detach(), view)
        ui, vi = u.floor().long(), v.floor().long()
        inside = (z > 0) & (ui >= 0) & (ui < view.width) & (vi >= 0) & (vi < view.height)
        uc, vc = ui.clamp(0, view.width - 1), vi.clamp(0, view.height - 1)
        outside = inside & ~target[vc, uc] & ~view.alpha_ignore[vc, uc]
        counts += outside.int()
    return counts


def _scene_extent(model: GaussianModel, views: list[View], kind: str) -> float:
    if kind == "background":
        centers = torch.stack([-(v.viewmat[:3, :3].T @ v.viewmat[:3, 3]) for v in views])
        return float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1
    means = model.params["means"].detach()
    return float((means - means.median(0).values).norm(dim=1).quantile(0.95)) * 1.1


@torch.no_grad()
def _grow(strategy: DefaultStrategy, params, optimizers, state, cap: int) -> tuple[int, int]:
    """DefaultStrategy growth (duplicate small / split large high-gradient Gaussians) with a hard cap:
    when the candidates would exceed `cap`, only the highest-gradient ones are grown."""
    grads = state["grad2d"] / state["count"].clamp_min(1)
    high = grads > strategy.grow_grad2d
    n = len(params["means"])
    small = torch.exp(params["scales"]).max(-1).values <= strategy.grow_scale3d * state["scene_scale"]
    budget = cap - n
    if budget <= 0:
        return 0, 0
    # Duplication adds one Gaussian, split adds one net Gaussian (2 replace 1).
    if int(high.sum()) > budget:
        threshold = torch.topk(grads[high], budget).values[-1]
        high = high & (grads >= threshold)
    is_dupli = high & small
    is_split = high & ~small
    n_dupli, n_split = int(is_dupli.sum()), int(is_split.sum())
    if n_dupli:
        duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
    is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=is_split.device)])
    if n_split:
        split(params=params, optimizers=optimizers, state=state, mask=is_split, revised_opacity=strategy.revised_opacity)
    return n_dupli, n_split


def _update_grad_state(state: dict, info: dict, n_views: int) -> None:
    """Accumulate normalized image-plane gradients exactly like gsplat 1.5.3
    DefaultStrategy._update_state (packed=False): grads [C,N,2] are scaled to NDC by the raster
    size (a crop behaves like a standalone training image) and by the number of views the step's
    loss is averaged over; each (camera, visible Gaussian) pair adds one gradient norm + count."""
    grads = info["means2d"].grad
    if grads is None:
        return
    n = info.get("n_trained", grads.shape[1])            # frozen Gaussians follow the trained slice
    grads = grads[:, :n].clone()
    grads[..., 0] *= info["width"] / 2.0 * n_views
    grads[..., 1] *= info["height"] / 2.0 * n_views
    if state["grad2d"] is None:
        state["grad2d"] = torch.zeros(n, device=grads.device)
        state["count"] = torch.zeros(n, device=grads.device)
    radii = info["radii"][:, :n]
    sel = (radii > 0).all(dim=-1) if radii.dim() == 3 else radii > 0   # [C,N]
    gs_ids = torch.where(sel)[1]
    state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
    state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))


def _group_losses(model: GaussianModel, group: list[View], cfg: Config, kind: str,
                  drop_mult: torch.Tensor | None, depth_w: float, frozen: GaussianModel | None = None):
    """Render a group of same-size views in one call; return per-view summed loss [C], per-view
    term dict ([C] tensors, zero where a term is inactive) and the gsplat info."""
    W, H = group[0].width, group[0].height
    viewmats = torch.stack([v.viewmat for v in group])
    Ks = torch.stack([v.K for v in group])
    rgb, depth, alpha, info = render_batch(model, viewmats, Ks, W, H, drop_mult=drop_mult, frozen=frozen,
                                           radius_clip=cfg.training_radius_clip)
    info["means2d"].retain_grad()
    pred = rgb.permute(0, 3, 1, 2)                                   # [C,3,H,W]
    image = torch.stack([v.image for v in group])
    m = torch.stack([v.loss_mask for v in group]).float()[:, None]    # [C,1,H,W]
    # Masked loss normalized by the raster area: a crop is treated as a standalone 3DGS image,
    # which keeps the gradient scale (and hence the densify threshold) standard.
    l1 = ((pred - image).abs() * m).mean((1, 2, 3))
    dssim = ((1 - ssim_map(pred * m, image * m, cfg.ssim_window)) * m).mean((1, 2, 3))
    terms = {"l1": l1, "dssim": dssim}
    loss = cfg.l1_weight * l1 + cfg.dssim_weight * dssim
    if kind == "instance" and cfg.alpha_weight > 0:
        valid = ~torch.stack([v.alpha_ignore for v in group])
        target = torch.stack([v.alpha_target for v in group]).float()
        count = valid.sum((1, 2))
        bce = F.binary_cross_entropy(alpha.clamp(1e-5, 1 - 1e-5), target, reduction="none")
        bce = (bce * valid).sum((1, 2)) / count.clamp_min(1)          # zero when no valid pixel
        terms["alpha"] = bce
        loss = loss + cfg.alpha_weight * bce
    if depth_w > 0 and any(v.sparse_depth is not None for v in group):
        target = torch.stack([v.sparse_depth if v.sparse_depth is not None else torch.zeros_like(v.alpha_target)
                              for v in group])
        valid = target > 0
        d = ((depth - target).abs() * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
        terms["depth"] = d
        loss = loss + depth_w * d
    return loss, terms, info


def _step_loss(model: GaussianModel, batch: list[View], cfg: Config, kind: str,
               drop_mult: torch.Tensor | None, depth_w: float, frozen: GaussianModel | None = None):
    """Mean loss over the step's views. Views are grouped by raster size and each group is one
    batched rasterization. Returns (loss, detached term sums, infos)."""
    groups: dict[tuple[int, int], list[View]] = {}
    for v in batch:
        groups.setdefault((v.width, v.height), []).append(v)
    total, sums, infos = 0.0, {}, []
    for group in groups.values():
        loss, terms, info = _group_losses(model, group, cfg, kind, drop_mult, depth_w, frozen)
        total = total + loss.sum() / len(batch)
        infos.append(info)
        for k, t in terms.items():
            sums[k] = sums.get(k, 0.0) + t.detach().sum() / len(batch)
    return total, sums, infos


@torch.no_grad()
def _structural_prune_mask(model: GaussianModel, views: list[View], cfg: Config, kind: str,
                           extra_prune: Callable[[torch.Tensor], torch.Tensor] | None) -> torch.Tensor:
    """Instance mask-projection prune OR the caller's extra test (e.g. outside the visual hull)."""
    bad = torch.zeros(len(model), dtype=torch.bool, device=model.device)
    if kind == "instance":
        bad |= mask_outside_counts(model, views, cfg.mask_prune_dilate_px) >= cfg.mask_prune_min_views
    if extra_prune is not None:
        extra = extra_prune(model.params["means"].detach())
        if extra.shape != bad.shape or extra.dtype != torch.bool:
            raise ValueError(f"extra_prune must return bool [{len(model)}], got {extra.dtype} {tuple(extra.shape)}")
        bad |= extra.to(bad.device)
    return bad


class BiasCorrectedSelectiveAdam:
    """gsplat's SelectiveAdam (fused kernel, updates only visible Gaussians) with Adam's bias
    correction restored. gsplat's kernel omits it, which makes the first few hundred steps 2-6x
    larger than torch Adam's; harmless in a 30k-step run, destructive in the short keyframe
    retrainings of Stage B (train PSNR 25.7 -> 17.3 in 6 steps). The correction factor
    sqrt(1 - b2^t) / (1 - b1^t) is applied to the step's learning rate (eps = 1e-15 is negligible);
    t counts optimizer steps (all Gaussians share it)."""

    def __new__(cls, param_groups, eps: float, betas: tuple[float, float]):
        from gsplat.optimizers import SelectiveAdam

        class _Opt(SelectiveAdam):
            _t = 0

            @torch.no_grad()
            def step(self, visibility):
                self._t += 1
                b1, b2 = self.param_groups[0]["betas"]
                f = (1 - b2 ** self._t) ** 0.5 / (1 - b1 ** self._t)
                base = [g["lr"] for g in self.param_groups]
                for g in self.param_groups: g["lr"] = g["lr"] * f
                try:
                    super().step(visibility)
                finally:
                    for g, lr in zip(self.param_groups, base): g["lr"] = lr

        return _Opt(param_groups, eps=eps, betas=betas)


def train_gaussians(model: GaussianModel, views: list[View], cfg: Config, kind: str = "instance",
                    log_path: str | Path | None = None, frozen: GaussianModel | None = None,
                    extra_prune: Callable[[torch.Tensor], torch.Tensor] | None = None) -> dict:
    """Optimize `model` in place. kind: "instance" (alpha BCE, per-instance cap) or "background".

    frozen: Gaussians rendered together with `model` in every training rasterization (depth-ordered
      compositing, e.g. person models in front of the background) but never optimized, dropped,
      densified or pruned.
    extra_prune: called with the trained means at every mask-prune event (every
      `mask_prune_every` steps during refinement) and at the final prune; True = remove."""
    if kind not in ("instance", "background"):
        raise ValueError(kind)
    if not views:
        raise ValueError("No training views")
    torch.manual_seed(cfg.seed)
    iterations = cfg.t0_iterations if kind == "instance" else cfg.bg_iterations
    cap = cfg.max_gaussians_per_instance if kind == "instance" else cfg.bg_max_gaussians
    extent = _scene_extent(model, views, kind)
    lrs = {"means": cfg.t0_lr_means * extent, "scales": cfg.t0_lr_scales, "quats": cfg.t0_lr_quats,
           "opacities": cfg.t0_lr_opacity, "sh0": cfg.t0_lr_sh0, "shN": cfg.t0_lr_shN}
    params = model.params
    # Selective Adam (Taming-3DGS, gsplat): only Gaussians visible in the step's views are updated;
    # its state layout (step, exp_avg, exp_avg_sq) is the one gsplat.strategy.ops expects.
    selective = bool(getattr(cfg, "selective_adam", True)) and params["means"].is_cuda
    if selective:
        optimizers = {k: BiasCorrectedSelectiveAdam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15, betas=(0.9, 0.999)) for k in PARAM_KEYS}
    else:
        optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15, fused=True)
                      for k in PARAM_KEYS}
    final_factor = cfg.t0_lr_means_final_factor
    means_gamma = final_factor ** (1.0 / max(iterations, 1))
    refine_stop = int(cfg.densify_until_fraction * iterations)
    strategy = DefaultStrategy(prune_opa=cfg.opacity_prune,
                               grow_grad2d=cfg.densify_grad_threshold * cfg.densify_gradient_multiplier,
                               refine_start_iter=cfg.densify_start_iter, refine_stop_iter=refine_stop,
                               refine_every=cfg.densify_every)
    state = strategy.initialize_state(scene_scale=extent)
    depth_until = cfg.depth_decay_fraction * iterations
    per_step = min(cfg.t0_views_per_step if kind == "instance" else cfg.bg_views_per_step, len(views))
    log = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w")
        log.write(json.dumps({"config": asdict(cfg), "kind": kind, "iterations": iterations, "cap": cap,
                              "extent": extent, "lrs": lrs, "views": [v.name for v in views],
                              "initial_gaussians": len(model)}) + "\n")
    history = []
    order = torch.randperm(len(views)).tolist()
    cursor = 0
    torch.cuda.synchronize()
    start = time.time()
    for step in range(iterations):
        drop = cfg.drop_gaussian_rate * step / iterations if cfg.drop_gaussian else 0.0  # progressive r_t = gamma t/T
        depth_w = cfg.depth_weight * max(0.0, 1.0 - step / depth_until) if depth_until > 0 else 0.0
        batch = []
        for _ in range(per_step):
            if cursor == len(order):
                order, cursor = torch.randperm(len(views)).tolist(), 0
            batch.append(views[order[cursor]])
            cursor += 1
        drop_mult = drop_multiplier(len(model), drop, model.device)  # one DropGaussian sample per step
        total, sums, infos = _step_loss(model, batch, cfg, kind, drop_mult, depth_w, frozen)
        log_scales = params["scales"]
        ratio = torch.exp(log_scales.max(-1).values - log_scales.min(-1).values)
        scale_reg = (ratio / cfg.scale_ratio_max - 1).clamp_min(0).mean()
        total = total + cfg.scale_reg_weight * scale_reg
        total.backward()
        if selective:
            n_tr = len(model)
            vis = torch.zeros(n_tr, dtype=torch.bool, device=params["means"].device)
            for info in infos:
                r = info["radii"][:, :n_tr]
                vis |= ((r > 0).all(-1) if r.dim() == 3 else r > 0).any(0)
        for opt in optimizers.values():
            opt.step(vis) if selective else opt.step()
            opt.zero_grad(set_to_none=True)
        for g in optimizers["means"].param_groups:
            g["lr"] *= means_gamma
        if step < refine_stop:
            for info in infos:
                _update_grad_state(state, info, per_step)
            if step > cfg.densify_start_iter and step % cfg.densify_every == 0:
                n_dupli, n_split = _grow(strategy, params, optimizers, state, cap)
                n_prune = strategy._prune_gs(params, optimizers, state, step)
                if cfg.mask_prune_every > 0 and step % cfg.mask_prune_every == 0:
                    bad = _structural_prune_mask(model, views, cfg, kind, extra_prune)
                    if bad.any() and int(bad.sum()) < len(model):
                        remove(params=params, optimizers=optimizers, state=state, mask=bad)
                        n_prune += int(bad.sum())
                state["grad2d"].zero_()
                state["count"].zero_()
                history.append({"step": step, "dupli": n_dupli, "split": n_split, "prune": n_prune, "n": len(model)})
        if log is not None and (step % 100 == 0 or step == iterations - 1):
            log.write(json.dumps({"step": step, "loss": float(total), **{k: float(v) for k, v in sums.items()}, "scale_reg": float(scale_reg),
                                  "drop_rate": drop, "depth_weight": depth_w, "n": len(model)}) + "\n")
            log.flush()
    # Final opacity prune (DefaultStrategy only prunes during refinement).
    with torch.no_grad():
        final_bad = (torch.sigmoid(params["opacities"]) < cfg.opacity_prune) | \
            _structural_prune_mask(model, views, cfg, kind, extra_prune)
        if final_bad.any() and int(final_bad.sum()) < len(model):
            remove(params=params, optimizers=optimizers, state=state, mask=final_bad)
    torch.cuda.synchronize()
    wall = time.time() - start
    stats = {"kind": kind, "iterations": iterations, "wall_time_s": wall, "it_per_s": iterations / max(wall, 1e-9),
             "final_gaussians": len(model), "densify_history": history, "extent": extent}
    if log is not None:
        log.write(json.dumps({"final": stats}) + "\n")
        log.close()
    return stats


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mse = F.mse_loss(pred, target)
    else:
        m = mask.float().expand_as(pred)
        mse = ((pred - target) ** 2 * m).sum() / m.sum().clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-12)))
