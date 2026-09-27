"""Leave-one-camera-out evaluation inside the 12 training cameras (development metric; never
reads the 24 held-out cameras). Person models are retrained without camera c (step_train with
exclude_cameras) and scored on c's person pixels, composited over the target background."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


def masked_psnr(render: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    err = ((render - target)[mask.astype(bool)] ** 2).mean()
    return float(-10 * np.log10(max(err, 1e-12)))


def alpha_iou(alpha: np.ndarray, mask: np.ndarray, threshold: float = .5) -> float:
    a = np.asarray(alpha) > threshold; b = np.asarray(mask).astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def floater_mass(opacity: np.ndarray, inside_hull: np.ndarray) -> float:
    return float(opacity[~inside_hull].sum() / max(opacity.sum(), 1e-12))


@torch.no_grad()
def loo_person_metrics(s, tag: str, camera: int) -> dict:
    """Person-region metrics on training camera `camera` for models under out/<tag>/."""
    import lpips
    from ring_init.gs.export import concat_models, load_ply
    from ring_init.gs.train import render, ssim
    dev = s.cfg.device; d = s.out / tag
    models = [load_ply(p, dev) for p in sorted(d.glob("instance_*/point_cloud.ply")) if p.parent.name != "instance_000"]
    persons = concat_models(models); cam = s.cameras[camera]; lab = s.labels()[camera]
    w2c = np.eye(4, dtype=np.float32); w2c[:3] = np.column_stack((cam.R, cam.t))
    rgb, _, alpha, _ = render(persons, torch.as_tensor(w2c, device=dev), torch.as_tensor(cam.K, dtype=torch.float32, device=dev), cam.width, cam.height)
    target = torch.as_tensor(s.images[camera], device=dev).float() / 255
    comp = (rgb + (1 - alpha[..., None]) * target).clamp(0, 1)   # persons over the true background
    mask = lab > 0
    ys, xs = np.nonzero(mask); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = torch.as_tensor(mask, device=dev)
    psnr = float(-10 * torch.log10(((comp - target) ** 2)[m].mean()))
    P, T = comp.permute(2, 0, 1)[None, :, y0:y1, x0:x1], target.permute(2, 0, 1)[None, :, y0:y1, x0:x1]
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    return {"camera": camera, "psnr_person": psnr, "ssim_person_bbox": float(ssim(P[0], T[0])), "lpips_person_bbox": float(lp(P*2-1, T*2-1).mean()),
            "iou": alpha_iou(alpha.cpu().numpy(), mask), "gaussians": int(len(persons))}


def run_loo(s, cameras: tuple[int, ...], tag_prefix: str = "loo") -> dict:
    """Retrain every person without each camera in turn (background not retrained) and score."""
    from dataclasses import replace
    from ring_init.stage_a import step_train
    results = {}
    for c in cameras:
        tag = f"{tag_prefix}_cam{c:02d}"
        s2 = type(s)(s.name, replace(s.cfg, background=False), s.force)
        step_train(s2, exclude_cameras=(c,), tag=tag)
        results[c] = loo_person_metrics(s2, tag, c)
        print(f"  LOO cam {c:02d}: {results[c]}")
    summary = {k: float(np.mean([r[k] for r in results.values()])) for k in ("psnr_person", "ssim_person_bbox", "lpips_person_bbox", "iou")}
    payload = {"cameras": {str(k): v for k, v in results.items()}, "mean": summary}
    (s.out / f"{tag_prefix}_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
