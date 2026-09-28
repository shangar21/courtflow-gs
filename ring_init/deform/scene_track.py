"""Stage B v2: QuickCapture-style online deformation of the WHOLE scene.

The live state is every Gaussian (persons + background). Per frame:
  1. deformation: control-point offsets (control points everywhere, denser on players) warp all
     Gaussians by rigid MLS; loss = full-frame L1 + D-SSIM + per-person SAM2 mask BCE + ARAP +
     temporal + triangulated mask-centroid anchor for players;
  2. bake: deformed Gaussians and advanced control points become the state;
  3. touch-up (between keyframes): a few iterations on SH DC + opacity;
  4. keyframe (every K frames): full 3DGS refinement of all Gaussians on the current 12 images
     (small position lr, capped densification), then Gaussians are re-bound to the controls."""
from __future__ import annotations
import time
from dataclasses import replace
import torch
import torch.nn.functional as F
from gsplat import rasterization
from ring_init.config import Config
from ring_init.deform import mls
from ring_init.deform.control_points import SceneControls, bind_gaussians
from ring_init.deform.regularizers import arap, cp_rotations_from_mls
from ring_init.deform.track import CameraFrame, eval_colors
from ring_init.gs.train import GaussianModel, View, ssim_map, train_gaussians


def densify_cap(n_trained: int, growth: float, base: int, cap: float, max_total: int = 0) -> int:
    """Maximum size of the trained model after one densifying refinement.

    ``growth`` limits a single keyframe relative to the current size. Without a cap that limit
    compounds (70 keyframes at 5% allow ~30x). ``cap > 0`` bounds the total at ``base * (1 + cap)``,
    where ``base`` is the same group's frame-0 size; ``max_total > 0`` bounds it at an absolute
    Gaussian count. Neither ever forces pruning."""
    limit = int(n_trained * (1 + growth))
    if cap > 0: limit = min(limit, max(int(base * (1 + cap)), n_trained))
    if max_total > 0: limit = min(limit, max(max_total, n_trained))
    return limit


class SceneTracker:
    def __init__(self, model: GaussianModel, ctrl: SceneControls, person_ids: list[int], cfg: Config):
        self.model, self.ctrl, self.cfg = model, ctrl, cfg
        self.person_ids = person_ids
        self.ids_t = torch.as_tensor(person_ids, device=ctrl.pos.device)
        self.spacing = ctrl.spacing
        self.gen = torch.Generator().manual_seed(cfg.seed); self.cam_cycle: list[int] = []
        # person index per control (-1 for background); used for per-person translations/anchor
        lookup = {k: i for i, k in enumerate(person_ids)}
        self.cp_person = torch.as_tensor([lookup.get(int(g), -1) for g in ctrl.group.tolist()], device=ctrl.pos.device)
        self.cp_pos0_mean = self.person_means(ctrl.pos)
        self.history: list[torch.Tensor] = [ctrl.pos.clone()]   # control positions per frame
        self.arap_neighborhood = torch.cat((torch.arange(len(ctrl.pos), device=ctrl.pos.device)[:, None], ctrl.graph), 1)
        self.arap_weights = torch.full(self.arap_neighborhood.shape, 1.0 / self.arap_neighborhood.shape[1],
                                       device=ctrl.pos.device, dtype=ctrl.pos.dtype)
        self._refresh_onehot()

    # ------------------------------------------------------------------ helpers
    def _refresh_onehot(self):
        ids = self.model.params["instance_ids"].long()
        self.onehot = (ids[:, None] == self.ids_t[None]).float()

    def person_means(self, pos: torch.Tensor) -> torch.Tensor:
        sel = self.cp_person >= 0
        out = torch.zeros(len(self.person_ids), 3, device=pos.device)
        return out.index_reduce_(0, self.cp_person[sel], pos[sel], "mean", include_self=False)

    def next_cameras(self, n: int, k: int) -> list[int]:
        if k >= n: return list(range(n))
        while len(self.cam_cycle) < k: self.cam_cycle += torch.randperm(n, generator=self.gen).tolist()
        chosen, self.cam_cycle = self.cam_cycle[:k], self.cam_cycle[k:]; return chosen

    def activated(self):
        p = self.model.params
        sh1 = p["shN"][:, :3] if p["shN"].shape[1] >= 3 else None
        return p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), p["sh0"], sh1

    def deform(self, offsets: torch.Tensor):
        means, quats, _, _, _, sh1 = self.activated()
        c = self.cfg
        return mls.deform(means.detach(), quats.detach(), None if sh1 is None else sh1.detach(), self.ctrl.pos, self.ctrl.nbr, self.ctrl.w,
                          offsets, None, False, 0.0, c.mls_svd_epsilon)

    def render(self, means, quats, sh1, cam: CameraFrame, with_instances: bool, opacities=None, sh0=None,
               subset: torch.Tensor | None = None, with_depth: bool = False, detach_fixed: bool = False,
               radius_clip: float = 0.0):
        _, _, scales, op, sh0_, _ = self.activated()
        op = op if opacities is None else opacities; sh0 = sh0_ if sh0 is None else sh0; onehot = self.onehot
        if subset is not None:
            means, quats, scales, op, sh0, onehot = means[subset], quats[subset], scales[subset], op[subset], sh0[subset], onehot[subset]
            sh1 = None if sh1 is None else sh1[subset]
        if detach_fixed:
            # Appearance is not an optimization variable during control-point tracking.  Keeping
            # its values while removing its autograd leaves preserves geometry gradients but
            # avoids allocating gradients for hundreds of thousands of fixed parameters.
            op, sh0 = op.detach(), sh0.detach()
        feats = eval_colors(sh0, sh1, means, cam.center)
        if with_instances: feats = torch.cat((feats, onehot), 1)
        out, alpha, _ = rasterization(means, quats, scales.detach(), op, feats, cam.viewmat[None], cam.K[None], cam.width, cam.height, sh_degree=None, packed=False,
                                      render_mode="RGB+ED" if with_depth else "RGB", radius_clip=radius_clip)
        rgb = out[0, ..., :3]; extra = out[0, ..., 3:]
        depth = extra[..., -1] if with_depth else None
        inst = (extra[..., :-1] if with_depth else extra) if with_instances else None
        if with_depth: return rgb, inst, alpha[0, ..., 0], depth
        return rgb, inst, alpha[0, ..., 0]

    def render_many(self, means, quats, sh1, cams: list[CameraFrame], with_instances: bool,
                    subset: torch.Tensor | None = None, with_depth: bool = False,
                    detach_fixed: bool = False, radius_clip: float = 0.0):
        """One padded gsplat call for several cameras of slightly different native sizes.

        Camera images occupy the top-left portion of the common canvas, so their intrinsics do
        not change; callers slice each render back to its native size before computing loss.
        """
        _, _, scales, op, sh0, _ = self.activated(); onehot = self.onehot
        if subset is not None:
            means, quats, scales, op, sh0, onehot = means[subset], quats[subset], scales[subset], op[subset], sh0[subset], onehot[subset]
            sh1 = None if sh1 is None else sh1[subset]
        if detach_fixed: op, sh0 = op.detach(), sh0.detach()
        width, height = max(c.width for c in cams), max(c.height for c in cams)
        feats = torch.stack([eval_colors(sh0, sh1, means, cam.center) for cam in cams])
        if with_instances: feats = torch.cat((feats, onehot[None].expand(len(cams), -1, -1)), -1)
        out, alpha, _ = rasterization(means, quats, scales.detach(), op, feats,
                                      torch.stack([c.viewmat for c in cams]), torch.stack([c.K for c in cams]), width, height,
                                      sh_degree=None, packed=False, render_mode="RGB+ED" if with_depth else "RGB", radius_clip=radius_clip)
        rgb, extra = out[..., :3], out[..., 3:]
        depth = extra[..., -1] if with_depth else None
        inst = (extra[..., :-1] if with_depth else extra) if with_instances else None
        return rgb, inst, alpha[..., 0], depth

    @torch.no_grad()
    def cache_background(self, cams: list[CameraFrame]) -> None:
        """Experiment 2: per-camera static background render (colour + expected depth), refreshed at keyframes."""
        means, quats, _, _, _, sh1 = self.activated(); bg = self.model.params["instance_ids"] == 0
        for cam in cams:
            rgb, _, a, d = self.render(means, quats, sh1, cam, False, subset=bg, with_depth=True)
            cam.bg_rgb, cam.bg_depth = rgb, torch.where(a > 0.5, d / a.clamp_min(1e-6), torch.full_like(d, float("inf")))
        self.dynamic = ~bg

    def photo_loss(self, rgb: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        l1 = (rgb - img).abs().mean()
        dssim = (1 - ssim_map(rgb.permute(2, 0, 1), img.permute(2, 0, 1), c.ssim_window)).mean()
        return c.l1_weight * l1 + c.dssim_weight * dssim

    # ------------------------------------------------------------------ per-frame deformation
    def deform_frame(self, cams: list[CameraFrame], anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, use_masks: bool = True) -> dict:
        """anchor = (displacement of each movable group's mean control position since frame 0 [I,3], valid [I], weight [I])."""
        c = self.cfg; sync = c.tracking_profile; timers = {k: 0.0 for k in ("mls_fwd", "render", "backward", "optimizer", "regularizers")}
        def mark(key, t0):
            if sync: torch.cuda.synchronize()
            timers[key] += time.perf_counter() - t0; return time.perf_counter()
        pos = self.ctrl.pos; hist = self.history
        vel = (hist[-1] - hist[-2]) if (c.tracking_warm_start and len(hist) >= 2) else torch.zeros_like(pos)
        pers = self.cp_person >= 0
        # per-person shared translation (players) + per-control offsets (everything)
        T0 = torch.zeros(len(self.person_ids), 3, device=pos.device).index_reduce_(0, self.cp_person[pers], vel[pers], "mean", include_self=False)
        if anchor is not None:   # re-anchor bulk motion on the triangulated mask centroids
            target = self.cp_pos0_mean + anchor[0]; cur = self.person_means(pos)
            T0 = torch.where(anchor[1][:, None], target - cur, T0)
        T = torch.nn.Parameter(T0); delta = torch.nn.Parameter(vel - torch.where(pers[:, None], T0[self.cp_person.clamp_min(0)], torch.zeros_like(vel)))
        bg_fixed = (self.ctrl.group == 0)[:, None] if c.v2_dynamic_only_deform else None
        def offsets():
            o = delta + torch.where(pers[:, None], T[self.cp_person.clamp_min(0)], torch.zeros_like(delta))
            return torch.where(bg_fixed, torch.zeros_like(o), o) if bg_fixed is not None else o
        opt = torch.optim.Adam([{"params": [T], "lr": c.m(c.tracking_lr_global_m)}, {"params": [delta], "lr": c.m(c.tracking_lr_translation_m)}], eps=c.tracking_adam_eps)
        history = []; n = c.early_stop_patience; it = 0
        for it in range(c.tracking_iterations):
            idx = self.next_cameras(len(cams), c.cameras_per_iteration)
            t0 = time.perf_counter(); off = offsets()
            means, quats, sh1 = self.deform(off); t0 = mark("mls_fwd", t0)
            loss = 0.0
            batch_cams = [cams[i] for i in idx]
            if c.tracking_batch_cameras:
                if c.v2_dynamic_only_deform:
                    rendered = self.render_many(means, quats, sh1, batch_cams, use_masks, subset=self.dynamic, with_depth=True,
                                                detach_fixed=True, radius_clip=c.tracking_radius_clip)
                else:
                    rendered = self.render_many(means, quats, sh1, batch_cams, use_masks, detach_fixed=True,
                                                radius_clip=c.tracking_radius_clip)
            for j, i in enumerate(idx):
                cam = cams[i]
                masked = use_masks and c.v2_mask_weight > 0 and cam.labels is not None
                if c.v2_dynamic_only_deform:   # only persons/ball rasterized, composited over the cached background with a depth test
                    if c.tracking_batch_cameras:
                        drgb, inst, da, dd = (v[j, :cam.height, :cam.width] if v is not None else None for v in rendered)
                    else:
                        drgb, inst, da, dd = self.render(means, quats, sh1, cam, masked, subset=self.dynamic, with_depth=True,
                                                          detach_fixed=True, radius_clip=c.tracking_radius_clip)
                    front = (dd / da.clamp_min(1e-6) <= cam.bg_depth) | (da < 1e-3)
                    rgb = torch.where(front[..., None], drgb + (1 - da[..., None]) * cam.bg_rgb, cam.bg_rgb)
                else:
                    if c.tracking_batch_cameras:
                        rgb, inst, _ = (v[j, :cam.height, :cam.width] if v is not None else None for v in rendered)
                    else:
                        rgb, inst, _ = self.render(means, quats, sh1, cam, masked, detach_fixed=True,
                                                    radius_clip=c.tracking_radius_clip)
                loss = loss + self.photo_loss(rgb, cam.image)
                if masked:
                    target = (cam.labels[..., None] == self.ids_t).float(); wgt = cam.valid_ids.float()[None, None, :].expand_as(target)
                    bce = F.binary_cross_entropy(inst.clamp(1e-5, 1 - 1e-5), target, weight=wgt, reduction="sum") / wgt.sum().clamp_min(1)
                    loss = loss + c.v2_mask_weight * bce
            loss = loss / len(idx); t0 = mark("render", t0)
            if c.arap_weight > 0:
                R = cp_rotations_from_mls(pos, off, self.ctrl.graph, self.arap_neighborhood, self.arap_weights)
                loss = loss + c.arap_weight * arap(pos, off, self.ctrl.graph, R)
            if c.temporal_weight > 0 and len(hist) >= 2:
                acc = (pos + off) - 2 * hist[-1] + hist[-2]
                loss = loss + c.temporal_weight * (acc.square().sum(-1) / self.spacing ** 2).mean()
            if anchor is not None and bool(anchor[1].any()):
                r = (self.person_means(pos + off) - (self.cp_pos0_mean + anchor[0])).norm(dim=-1) / c.m(c.tracking_anchor_scale_m)
                hub = F.huber_loss(r, torch.zeros_like(r), delta=1.0, reduction="none")
                loss = loss + (anchor[2] * hub)[anchor[1]].sum() / anchor[1].sum()
            t0 = mark("regularizers", t0)
            opt.zero_grad(set_to_none=True); loss.backward(); t0 = mark("backward", t0)
            opt.step(); mark("optimizer", t0)
            history.append(loss.detach())
            if len(history) >= 2 * n and it >= c.tracking_min_iterations and it % 5 == 0:
                h = torch.stack(history[-2 * n:])
                if float(h[:n].mean()) - float(h[n:].mean()) < c.early_stop_delta * float(h[:n].mean()): break
        # bake: deformed Gaussians and advanced controls become the state
        with torch.no_grad():
            off = offsets(); means, quats, sh1 = self.deform(off); p = self.model.params
            p["means"].data.copy_(means); p["quats"].data.copy_(quats)
            if sh1 is not None: p["shN"].data[:, :3] = sh1
            self.ctrl.pos = pos + off; self.history.append(self.ctrl.pos.clone())
            self.history = self.history[-3:]
        w = min(len(history), n)
        return {**timers, "iterations": it + 1, "first_loss": float(torch.stack(history[:w]).mean()), "final_loss": float(torch.stack(history[-w:]).mean())}

    # ------------------------------------------------------------------ appearance touch-up
    def touch_up(self, cams: list[CameraFrame]) -> float:
        c = self.cfg
        if c.v2_touchup_iterations <= 0: return 0.0
        t0 = time.time(); p = self.model.params
        opt = torch.optim.Adam([{"params": [p["sh0"]], "lr": c.t0_lr_sh0}, {"params": [p["opacities"]], "lr": c.t0_lr_opacity * 0.2}], eps=1e-15)
        for _ in range(c.v2_touchup_iterations):
            for i in self.next_cameras(len(cams), c.cameras_per_iteration):
                means, quats, _, _, _, sh1 = self.activated()
                rgb, _, _ = self.render(means.detach(), quats.detach(), None if sh1 is None else sh1.detach(), cams[i], False,
                                        opacities=torch.sigmoid(p["opacities"]), sh0=p["sh0"])
                (self.photo_loss(rgb, cams[i].image) / c.cameras_per_iteration).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        for k in ("means", "quats", "scales", "shN"): p[k].grad = None
        return time.time() - t0

    # ------------------------------------------------------------------ keyframe retraining
    def focus_views(self, views: list[View], pad_px: int) -> list[View]:
        """Experiment 5: crop each keyframe view to the (tile-aligned) box around all dynamic Gaussians."""
        from ring_init.gs.train import crop_view
        dyn = self.model.params["means"][self.model.params["instance_ids"] != 0].detach(); out = []
        for v in views:
            pc = dyn @ v.viewmat[:3, :3].T + v.viewmat[:3, 3]; ok = pc[:, 2] > 0
            uv = (pc[ok] @ v.K.T); uv = uv[:, :2] / uv[:, 2:]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < v.width) & (uv[:, 1] >= 0) & (uv[:, 1] < v.height); uv = uv[inside]
            if len(uv) == 0: out.append(v); continue
            lo = (uv.min(0).values - pad_px).floor().int().tolist(); hi = (uv.max(0).values + pad_px).ceil().int().tolist()
            out.append(crop_view(v, (max(lo[0], 0), max(lo[1], 0), min(hi[0], v.width), min(hi[1], v.height))))
        return out

    def refine(self, views: list[View], log_path, iterations: int, views_per_step: int,
               focus: bool = False, densify: bool = False, growth: float = 0.0,
               dynamic_only: bool = False) -> dict:
        """Refine the live Gaussian state against the current images.

        Keyframes permit limited densification; inter-keyframe refinements deliberately do
        not, so the control binding and model size remain stable from frame to frame.  Focused
        refinement can additionally keep the static group frozen: the dynamic model is trained
        while the background is still depth-composited by ``train_gaussians``.
        """
        c = self.cfg; t0 = time.time(); n0 = len(self.model)
        if focus: views = self.focus_views(views, c.v2_keyframe_focus_pad_px)
        trained, frozen, base = self.model, None, self.n_canon
        if dynamic_only:
            from ring_init.gs.train import PARAM_KEYS
            bg = self.model.params["instance_ids"] == 0
            if not bool(bg.any()) or bool(bg.all()):
                raise RuntimeError("dynamic-only refinement requires both background and dynamic Gaussians")
            trained = GaussianModel({k: self.model.params[k][~bg] for k in PARAM_KEYS}, self.model.params["instance_ids"][~bg])
            frozen = GaussianModel({k: self.model.params[k][bg] for k in PARAM_KEYS}, self.model.params["instance_ids"][bg])
            base = self.n_canon_dynamic
        kcfg = replace(c, opacity_prune=0.0 if c.v2_keyframe_protect else c.opacity_prune, bg_iterations=iterations, bg_views_per_step=views_per_step,
                       t0_lr_means=c.v2_keyframe_lr_means, t0_lr_means_final_factor=1.0, drop_gaussian=False, depth_weight=0.0,
                       densify_start_iter=c.v2_keyframe_densify_start if densify else iterations + 1,
                       densify_every=c.v2_keyframe_densify_every, densify_until_fraction=c.v2_keyframe_densify_until if densify else 0.0,
                       bg_max_gaussians=densify_cap(len(trained), growth, base, c.v2_growth_cap,
                                                    max(c.v2_max_gaussians - len(frozen), 1) if frozen is not None and c.v2_max_gaussians > 0 else c.v2_max_gaussians))
        stats = train_gaussians(trained, views, kcfg, "background", log_path, frozen=frozen)
        if frozen is not None:
            # Restore the normal background-first layout after updating only the dynamic slice.
            from ring_init.gs.export import concat_models
            self.model = concat_models([frozen, trained])
        with torch.no_grad():
            self.ctrl.nbr, self.ctrl.w = bind_gaussians(self.model.params["means"], self.model.params["instance_ids"].long(), self.ctrl.pos, self.ctrl.group, self.ctrl.sigma, c.gaussian_control_knn)
        self._refresh_onehot()
        for k in self.model.params: self.model.params[k].grad = None
        return {"refine_s": time.time() - t0, "gaussians_before": n0, "gaussians_after": len(self.model), "refine_final_loss": stats.get("final_loss")}

    def keyframe(self, views: list[View], log_path, focus: bool = False) -> dict:
        result = self.refine(views, log_path, self.cfg.v2_keyframe_iterations,
                             self.cfg.v2_keyframe_views_per_step, focus, densify=True,
                             growth=self.cfg.v2_keyframe_growth, dynamic_only=self.cfg.v2_keyframe_dynamic_only)
        return {"keyframe_s": result.pop("refine_s"), "gaussians_before": result.pop("gaussians_before"),
                "gaussians_after": result.pop("gaussians_after"), "keyframe_final_loss": result.pop("refine_final_loss")}


    @torch.no_grad()
    def reprompt_labels(self, cams: list[CameraFrame], images_full: list, refiner, min_weight: float = 0.3, pad_px: int = 4) -> list[torch.Tensor]:
        """Keyframe masks without video propagation: project the TRACKED people into every view, prompt
        SAM2 with their boxes (+ centroid point) on the full-resolution image, and assign overlapping
        pixels to the person with the larger rendered weight. Returns tracking-resolution labels."""
        import numpy as np
        means, quats, _, _, _, sh1 = self.activated(); labels = []
        for cam, img in zip(cams, images_full):
            _, inst, _ = self.render(means, quats, sh1, cam, True)                  # [h,w,I]
            s = img.shape[1] / cam.width; boxes, points, cols = [], [], []
            for j, k in enumerate(self.person_ids):
                if not bool(cam.valid_ids[j]): continue                                # e.g. the ball
                ys, xs = torch.nonzero(inst[..., j] > min_weight, as_tuple=True)
                if len(xs) < 4: continue
                boxes.append([float(xs.min()) * s - pad_px, float(ys.min()) * s - pad_px, float(xs.max() + 1) * s + pad_px, float(ys.max() + 1) * s + pad_px])
                points.append([float(xs.float().mean()) * s, float(ys.float().mean()) * s]); cols.append(j)
            lab = torch.zeros((cam.height, cam.width), dtype=torch.long, device=cam.viewmat.device)
            if boxes:
                refiner.set_image(img)
                masks, _ = refiner.masks(np.asarray(boxes), np.asarray(points))
                m = torch.nn.functional.interpolate(torch.as_tensor(masks, device=lab.device).float()[None], size=(cam.height, cam.width), mode="area")[0] > 0.5
                score = torch.where(m, inst[..., cols].permute(2, 0, 1), torch.full_like(m, -1.0, dtype=torch.float32))
                best = score.max(0)
                lab = torch.where(best.values >= 0, self.ids_t[torch.as_tensor(cols, device=lab.device)][best.indices], lab)
            labels.append(lab)
        return labels

    # ------------------------------------------------------------------ checkpoints
    def checkpoint_state(self) -> dict:
        """Everything the tracker changes while running; the rest is rebuilt from the canonical scene."""
        return {"params": {k: v.detach().cpu() for k, v in self.model.params.items()},
                "ctrl": {k: getattr(self.ctrl, k).detach().cpu() for k in ("pos", "nbr", "w")},
                "history": [h.detach().cpu() for h in self.history],
                "gen": self.gen.get_state(), "cam_cycle": list(self.cam_cycle)}

    def restore_state(self, state: dict) -> None:
        from ring_init.gs.train import PARAM_KEYS
        dev = self.ctrl.pos.device; p = state["params"]
        self.model = GaussianModel({k: p[k].to(dev) for k in PARAM_KEYS}, p["instance_ids"].to(dev))
        for k, v in state["ctrl"].items(): setattr(self.ctrl, k, v.to(dev))
        self.history = [h.to(dev) for h in state["history"]]
        self.gen.set_state(state["gen"]); self.cam_cycle = list(state["cam_cycle"])
        self._refresh_onehot()

    # ------------------------------------------------------------------ recovery
    def remember_canonical(self) -> None:
        """Keep each movable group's canonical Gaussians and control positions (for re-initialization)."""
        p = self.model.params; ids = p["instance_ids"].long()
        self.canon = {k: {n: v[ids == k].detach().clone() for n, v in p.items()} for k in self.person_ids}
        self.canon_ctrl = self.ctrl.pos.detach().clone()
        self.n_canon, self.n_canon_dynamic = len(ids), int((ids != 0).sum())

    @torch.no_grad()
    def person_iou(self, cams: list[CameraFrame]) -> torch.Tensor:
        """Per movable group: IoU of rendered person weight (> 0.5) vs its mask, pooled over views where it is labelled."""
        means, quats, _, _, _, sh1 = self.activated()
        inter = torch.zeros(len(self.person_ids), device=means.device); union = torch.zeros_like(inter)
        for cam in cams:
            if cam.labels is None: continue
            _, inst, _ = self.render(means, quats, sh1, cam, True)
            a = inst > 0.5; g = cam.labels[..., None] == self.ids_t
            v = cam.valid_ids.float()
            inter += (a & g).sum((0, 1)).float() * v; union += (a | g).sum((0, 1)).float() * v
        return torch.where(union > 0, inter / union.clamp_min(1), torch.full_like(inter, float("nan")))

    @torch.no_grad()
    def reinit_person(self, k: int, displacement: torch.Tensor) -> None:
        """Replace group k's Gaussians by its canonical model moved rigidly by `displacement` (frame-0
        pose at the current triangulated position) and reset its control points and history."""
        from ring_init.deform.control_points import bind_gaussians
        p = self.model.params; ids = p["instance_ids"].long(); keep = ids != k
        new = {n: torch.cat((v[keep], self.canon[k][n] + (displacement if n == "means" else 0))) for n, v in p.items()}
        for n in list(p.keys()): p[n] = torch.nn.Parameter(new[n], requires_grad=n != "instance_ids")
        sel = self.ctrl.group == k
        self.ctrl.pos[sel] = self.canon_ctrl[sel] + displacement
        for h in self.history: h[sel] = self.ctrl.pos[sel]
        self.ctrl.nbr, self.ctrl.w = bind_gaussians(p["means"], p["instance_ids"].long(), self.ctrl.pos, self.ctrl.group, self.ctrl.sigma, self.cfg.gaussian_control_knn)
        self._refresh_onehot()
