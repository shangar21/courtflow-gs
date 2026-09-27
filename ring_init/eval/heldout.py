"""Held-out evaluation of the frozen Stage A scene. This is the ONLY module that reads the 24
held-out cameras; run it after every model is frozen."""
from __future__ import annotations
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ring_init.config import Config
from ring_init.io.calib import Camera


def load_all_cameras(sparse: str | Path) -> dict[int, Camera]:
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse)); result = {}
    for im in rec.images.values():
        cam = rec.camera(im.camera_id); T = im.cam_from_world().matrix()
        view = int(Path(im.name).stem.split("_")[-1])
        result[view] = Camera(im.name, cam.calibration_matrix(), np.zeros(0), T[:, :3], T[:, 3], cam.width, cam.height)
    return result


def _viewmat(cam: Camera, device: str) -> torch.Tensor:
    w2c = np.eye(4, dtype=np.float32); w2c[:3] = np.column_stack((cam.R, cam.t)); return torch.as_tensor(w2c, device=device)


def person_roi(occupancy: dict[str, np.ndarray], cam: Camera, radius_px: int) -> np.ndarray:
    """Region around every person: dilated projection of the occupancy voxels (geometry only)."""
    from ring_init.masks.instances import splat_silhouette
    roi = np.zeros((cam.height, cam.width), bool)
    for pts in occupancy.values(): roi |= splat_silhouette(pts, cam, radius_px)[0]
    return roi


def evaluate(scene: str, cfg: Config, canonical: Path | None = None, out_name: str = "eval") -> dict:
    from ring_init.gs.export import concat_models, load_ply
    from ring_init.gs.train import render, ssim
    from ring_init.geom.hull import inside_hull
    import lpips
    if not cfg.eval_sparse or not cfg.eval_images: raise RuntimeError("config.eval_sparse and config.eval_images are required for held-out evaluation")
    out = Path(cfg.out_root) / scene; canonical = canonical or out / "canonical"; dev = cfg.device
    manifest = json.loads((canonical / "manifest.json").read_text())
    models = {e["instance_id"]: load_ply(Path(e["model"]), dev) for e in manifest["instances"]}
    scene_model = concat_models([models[k] for k in sorted(models)])
    persons = concat_models([models[k] for k in sorted(models) if k != 0]) if len(models) > 1 else None
    cams = load_all_cameras(cfg.eval_sparse); train_views = list(cfg.eval_training_views)
    heldout = sorted(v for v in cams if v not in train_views)
    occ = dict(np.load(out / "instances" / "occupancy.npz"))
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    # Held-out person masks for IoU: detector + SAM2 on held-out images (evaluation only).
    from ring_init.masks.segment import SAM2Refiner, torchvision_person_detector
    detector = torchvision_person_detector(cfg.detector, cfg.detector_min_size, cfg.detector_max_size, cfg.detector_score_threshold, cfg.detector_max_detections, dev)
    refiner = SAM2Refiner(cfg.sam2_checkpoint, cfg.sam2_config, dev)
    ed = out / out_name; ed.mkdir(parents=True, exist_ok=True); rows = []
    for view in train_views + heldout:
        cam = cams[view]; path = Path(cfg.eval_images) / cam.name
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        target = torch.as_tensor(rgb, device=dev).float().div(255)
        vm = _viewmat(cam, dev); K = torch.as_tensor(cam.K, dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred, _, _, _ = render(scene_model, vm, K, cam.width, cam.height)
            pred = pred.clamp(0, 1)
            p_alpha = render(persons, vm, K, cam.width, cam.height)[2] if persons is not None else torch.zeros_like(pred[..., 0])
        roi = person_roi(occ, cam, 6); roi_t = torch.as_tensor(roi, device=dev)
        mse = ((pred - target) ** 2).mean(); psnr = float(-10 * torch.log10(mse))
        pm = ((pred - target) ** 2)[roi_t].mean(); psnr_person = float(-10 * torch.log10(pm))
        P, T = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
        with torch.no_grad():
            s_full = float(ssim(P[0], T[0])); lp_full = float(lp(P * 2 - 1, T * 2 - 1).mean())
            ys, xs = np.nonzero(roi); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            Pc, Tc = P[..., y0:y1, x0:x1], T[..., y0:y1, x0:x1]
            s_person = float(ssim(Pc[0], Tc[0])); lp_person = float(lp(Pc * 2 - 1, Tc * 2 - 1).mean())
        # IoU: rendered person alpha vs detector/SAM2 person masks, both restricted to the court ROI.
        boxes = detector(rgb); refiner.set_image(rgb)
        masks = refiner.masks(boxes)[0] if len(boxes) else np.zeros((0, cam.height, cam.width), bool)
        gt = (masks.any(0) if len(masks) else np.zeros(roi.shape, bool)) & roi
        pa = (p_alpha.cpu().numpy() > 0.5) & roi
        iou = float((gt & pa).sum() / max((gt | pa).sum(), 1))
        split = "train" if view in train_views else "heldout"
        rows.append({"view": view, "split": split, "psnr": psnr, "ssim": s_full, "lpips": lp_full, "psnr_person_roi": psnr_person, "ssim_person_bbox": s_person, "lpips_person_bbox": lp_person, "person_iou": iou})
        print(f"  {split:7s} view {view:02d}: PSNR {psnr:.2f}  person-ROI PSNR {psnr_person:.2f}  SSIM {s_full:.3f}  LPIPS {lp_full:.3f}  IoU {iou:.3f}")
        if view in (1, 13, 26) or view in train_views[:1]:
            save_comparison(pred, target, roi, ed / f"view_{view:02d}_{split}.jpg")
    del detector, refiner; torch.cuda.empty_cache()
    # Floater mass: opacity outside each instance's own (1-voxel dilated) hull.
    floaters = {}
    for k, m in models.items():
        if k == 0: continue
        h = np.load(out / "hulls" / f"instance_{k:03d}.npz")
        inside = inside_hull(m.params["means"].detach().cpu().numpy().astype(np.float64), h["occupied"].astype(np.float64), float(h["voxel"]), 1)
        op = torch.sigmoid(m.params["opacities"]).detach().cpu().numpy()
        floaters[k] = float(op[~inside].sum() / max(op.sum(), 1e-12))
    # Render FPS for the composed scene at full resolution (held-out camera 1).
    cam = cams[heldout[0]]; vm = _viewmat(cam, dev); K = torch.as_tensor(cam.K, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for _ in range(5): render(scene_model, vm, K, cam.width, cam.height)
        torch.cuda.synchronize(); t = time.time(); n = 50
        for _ in range(n): render(scene_model, vm, K, cam.width, cam.height)
        torch.cuda.synchronize(); fps = n / (time.time() - t)
    stats = json.loads((out / "train" / "stats.json").read_text()) if (out / "train" / "stats.json").is_file() else {}
    timings = json.loads((out / "timings.json").read_text()) if (out / "timings.json").is_file() else {}

    def mean(split, key): return float(np.mean([r[key] for r in rows if r["split"] == split]))
    summary = {split: {key: mean(split, key) for key in ("psnr", "ssim", "lpips", "psnr_person_roi", "ssim_person_bbox", "lpips_person_bbox", "person_iou")} for split in ("train", "heldout")}
    result = {"summary": summary, "baseline_heldout_psnr_db": cfg.baseline_heldout_psnr_db, "views": rows, "floater_mass": floaters,
              "gaussians": {str(k): len(m) for k, m in models.items()}, "total_gaussians": len(scene_model), "render_fps": fps,
              "train_seconds": {k: v.get("wall_time_s") for k, v in stats.items()}, "pipeline_seconds": timings}
    (ed / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (ed / "report.md").write_text(report_markdown(result)); print(report_markdown(result))
    return result


def save_comparison(pred: torch.Tensor, target: torch.Tensor, roi: np.ndarray, path: Path) -> None:
    p = (pred.cpu().numpy() * 255).astype(np.uint8); t = (target.cpu().numpy() * 255).astype(np.uint8)
    ys, xs = np.nonzero(roi); pad = 40
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    full = np.concatenate((cv2.resize(t, (960, int(960 * t.shape[0] / t.shape[1]))), cv2.resize(p, (960, int(960 * t.shape[0] / t.shape[1])))), 1)
    zt, zp = t[y0:y1, x0:x1], p[y0:y1, x0:x1]; h = int(960 * zt.shape[0] / zt.shape[1])
    zoom = np.concatenate((cv2.resize(zt, (960, h)), cv2.resize(zp, (960, h))), 1)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate((full, zoom), 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def report_markdown(r: dict) -> str:
    s = r["summary"]; lines = ["# Stage A evaluation (frame 0)", "",
         "| split | PSNR | SSIM | LPIPS | person-ROI PSNR | person SSIM | person LPIPS | person IoU |", "|---|---|---|---|---|---|---|---|"]
    for split in ("train", "heldout"):
        x = s[split]; lines.append(f"| {split} | {x['psnr']:.2f} | {x['ssim']:.3f} | {x['lpips']:.3f} | {x['psnr_person_roi']:.2f} | {x['ssim_person_bbox']:.3f} | {x['lpips_person_bbox']:.3f} | {x['person_iou']:.3f} |")
    lines += ["", f"Baseline global t0 held-out PSNR: {r['baseline_heldout_psnr_db']:.2f} dB", "",
              f"Gaussians: {r['total_gaussians']:,} total; render {r['render_fps']:.1f} FPS full-frame", "",
              "| instance | gaussians | floater mass | train s |", "|---|---|---|---|"]
    for k, n in sorted(r["gaussians"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {k} | {n:,} | {r['floater_mass'].get(int(k), r['floater_mass'].get(k, float('nan'))) if k != '0' else float('nan'):.4f} | {r['train_seconds'].get(k) or float('nan'):.1f} |")
    lines += ["", "Pipeline seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in r["pipeline_seconds"].items()), ""]
    return "\n".join(lines)
