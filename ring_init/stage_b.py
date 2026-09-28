"""Stage B: online per-frame deformation of the frozen Stage A canonical models."""
from __future__ import annotations
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from ring_init.config import Config
from ring_init.stage_a import Scene, scaled


def source_view(cam) -> int:
    return int(Path(cam.name).stem.split("_")[-1])


def _extract_job(args) -> tuple[int, int]:
    from ring_init.io.frames import ViewPreprocessor, extract_view
    view, cfg_dict, out_dir, start, end, stride, ext = args
    cfg = Config(**cfg_dict)
    pre = ViewPreprocessor(view, Path(cfg.distorted_cameras_txt), Path(cfg.distorted_images_txt), Path(cfg.eval_sparse), Path(cfg.photometric_json))
    if cfg.frame_gpu_preprocess: pre.to_device(cfg.device)
    return view, extract_view(pre, Path(cfg.videos_dir) / f"view_{view:03d}.mp4", Path(out_dir), start, end, stride, ext, cfg.frame_jpeg_quality)


def step_frames(s: Scene, start: int, end: int) -> None:
    """Training cameras: every frame -> frames/cam_XX/. Held-out cameras (evaluation only):
    every eval_frame_stride-th frame -> eval_frames/view_XXX/."""
    from dataclasses import asdict
    cfg = s.cfg
    for key in ("videos_dir", "distorted_cameras_txt", "distorted_images_txt", "photometric_json", "eval_sparse"):
        if not getattr(cfg, key): raise RuntimeError(f"config.{key} is required for Stage B frame extraction")
    t = time.time(); train_views = [source_view(c) for c in s.cameras]
    jobs = [(v, asdict(cfg), str(s.out / "frames" / f"cam_{i:02d}"), start, end, 1, cfg.frame_format) for i, v in enumerate(train_views)]
    heldout = [v for v in range(36) if v not in train_views]
    # Held-out frames are scoring targets, so they stay lossless whatever the training format.
    jobs += [(v, asdict(cfg), str(s.out / "eval_frames" / f"view_{v:03d}"), start, end, cfg.eval_frame_stride, "png") for v in heldout]
    # CUDA cannot be re-initialized in a forked child of this (CUDA-using) process.
    import multiprocessing as mp
    ctx = mp.get_context("spawn") if cfg.frame_gpu_preprocess else None
    with ProcessPoolExecutor(cfg.frame_workers, mp_context=ctx) as pool:
        for view, n in pool.map(_extract_job, jobs): print(f"  frames view {view:03d}: {n} new")
    timings = json.loads(s.timings_path.read_text()) if s.timings_path.is_file() else {}
    timings["stage_b_frames"] = time.time() - t; s.timings_path.write_text(json.dumps(timings, indent=2) + "\n")


def frame_image(s: Scene, cam: int, frame: int) -> np.ndarray:
    import cv2
    from ring_init.io.frames import frame_file
    path = frame_file(s.out / "frames" / f"cam_{cam:02d}", frame)
    im = cv2.imread(str(path))
    if im is None: raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sam2_worker_count(requested: int, cameras: int, frames: int, available_bytes: int, image_size: int = 1024) -> int:
    """Parallel SAM2 processes that fit in host memory. SAM2 buffers each camera's whole clip on
    the CPU as float32 (frames x 3 x image_size^2); allow 25% plus 1.5 GiB per process on top."""
    per_worker = frames * 3 * image_size ** 2 * 4 * 1.25 + 1.5 * 2 ** 30
    return max(1, min(requested, cameras, int(available_bytes // per_worker)))


def _available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"): return int(line.split()[1]) * 1024
    return 0


_SAM2 = None


def _init_sam2(config: str, checkpoint: str, device: str) -> None:
    global _SAM2
    from sam2.build_sam import build_sam2_video_predictor
    _SAM2 = build_sam2_video_predictor(config, checkpoint, device=device)


def _mask_job(args) -> tuple[int, dict]:
    from ring_init.masks.video import propagate_camera
    ci, frame_dir, frames, labels, out, min_area = args
    return ci, propagate_camera(_SAM2, Path(frame_dir), frames, labels, Path(out), min_area)


def step_video_masks(s: Scene, start: int, end: int, name: str = "video_masks") -> None:
    """Propagate the Stage A frame-0 instance labels through time per training camera.
    Cameras are independent, so up to ``sam2_workers`` processes propagate them concurrently."""
    cfg = s.cfg; d = s.dir(name); labels = s.labels(); frames = list(range(start, end)); report = {}
    if start != 0: raise ValueError("Mask propagation is prompted with the Stage A frame-0 labels; start must be 0.")
    jobs = []
    for ci in range(len(labels)):
        out = d / f"cam_{ci:02d}"
        if out.is_dir() and len(list(out.glob("*.png"))) >= len(frames) and "video_masks" not in s.force: continue
        jobs.append((ci, str(s.out / "frames" / f"cam_{ci:02d}"), frames, labels[ci], str(out), cfg.min_mask_area_px))
    workers = sam2_worker_count(cfg.sam2_workers, len(jobs), len(frames), _available_memory()) if jobs else 0
    t = time.time()
    if workers > 1:
        import multiprocessing as mp
        print(f"  video masks: {len(jobs)} cameras on {workers} SAM2 processes")
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_sam2,
                                 initargs=(cfg.sam2_config, cfg.sam2_checkpoint, cfg.device)) as pool:
            for ci, r in pool.map(_mask_job, jobs):
                report[ci] = r; print(f"  video masks cam {ci:02d}: {r}")
    elif jobs:
        _init_sam2(cfg.sam2_config, cfg.sam2_checkpoint, cfg.device)
        for job in jobs:
            ci, r = _mask_job(job); report[ci] = r; print(f"  video masks cam {ci:02d}: {r}")
    if jobs:
        timings = json.loads(s.timings_path.read_text()) if s.timings_path.is_file() else {}
        timings["stage_b_video_masks"] = time.time() - t; s.timings_path.write_text(json.dumps(timings, indent=2) + "\n")
    prev = json.loads((d / "report.json").read_text()) if (d / "report.json").is_file() else {}
    prev.update({str(k): v for k, v in report.items()}); (d / "report.json").write_text(json.dumps(prev, indent=2) + "\n")


# --------------------------------------------------------------------------- tracking
def near_plane(s: Scene) -> float:
    """Data-derived near plane: a fraction of the nearest triangulated structure seen by any
    training camera (0.1th depth percentile of the fused background seeds). Background Gaussians
    closer than that are unsupported floaters that only show up from novel viewpoints."""
    from ring_init.stage_a import _read_ply
    pts = _read_ply(s.out / "fuse" / "background.ply")[0]; nearest = []
    for cam in s.cameras:
        _, z = cam.project(pts); nearest.append(np.percentile(z[z > 0], 0.1))
    return s.cfg.render_near_plane_fraction * float(min(nearest))


def _camera_frame(cam, bg_model, device: str, scale: float = 1.0, near: float = 0.0):
    """Camera (optionally rescaled; pixel-centre convention preserved) + static background render
    (background Gaussians closer than `near` to this camera are skipped)."""
    import torch
    from ring_init.deform.track import CameraFrame
    from ring_init.gs.train import render
    w2c = np.eye(4, dtype=np.float32); w2c[:3] = np.column_stack((cam.R, cam.t))
    W, H = int(round(cam.width * scale)), int(round(cam.height * scale))
    Kn = cam.K.copy(); Kn[:2, :2] *= scale; Kn[:2, 2] = (cam.K[:2, 2] + 0.5) * scale - 0.5
    vm = torch.as_tensor(w2c, device=device); K = torch.as_tensor(Kn, dtype=torch.float32, device=device)
    with torch.no_grad():
        if bg_model is not None and near > 0:
            from ring_init.gs.train import GaussianModel
            z = bg_model.params["means"] @ vm[2, :3] + vm[2, 3]
            keep = z > near
            bg_model = GaussianModel({k: v[keep] for k, v in bg_model.params.items() if k != "instance_ids"}, bg_model.params["instance_ids"][keep])
        bg = render(bg_model, vm, K, W, H)[0].clamp(0, 1) if bg_model is not None else torch.zeros((H, W, 3), device=device)
    return CameraFrame(vm, K, W, H, torch.as_tensor(cam.center, dtype=torch.float32, device=device), bg)


def load_canonical(s: Scene, name: str = "canonical"):
    from ring_init.gs.export import load_ply
    manifest = json.loads((s.out / name / "manifest.json").read_text())
    return {e["instance_id"]: load_ply(e["model"], s.cfg.device) for e in manifest["instances"]}


def step_track(s: Scene, start: int, end: int, tag: str = "stage_b", exclude_cameras: tuple[int, ...] = (), masks: str = "video_masks") -> dict:
    """Online tracking of frames start+1 .. end-1 (frame `start` = canonical rest state)."""
    import torch
    from ring_init.deform.control_points import build_topology
    import cv2
    from ring_init.deform.track import Persons, Tracker, identity_state, mask_centroid_anchor, warm_start
    from ring_init.masks.video import load_video_labels
    cfg = scaled(s.cfg, s); dev = cfg.device; d = s.dir(tag); cfg.save(d / "config.json")
    models = load_canonical(s); persons = Persons.from_models(models)
    t0 = time.time()
    topo = build_topology([(k, models[k].params["means"].detach()) for k in persons.ids], cfg.control_points, cfg.gaussian_control_knn, cfg.control_knn, cfg.mls_sigma_knn)
    setup_s = time.time() - t0
    torch.save({"rest": topo.rest, "gaussian_neighbors": topo.gaussian_neighbors, "gaussian_weights": topo.gaussian_weights, "graph_neighbors": topo.graph_neighbors, "cp_instance": topo.cp_instance}, d / "topology.pt")
    cams = [_camera_frame(c, models.get(0), dev, cfg.tracking_scale) for c in s.cameras]
    full_cams = [_camera_frame(c, models.get(0), dev) for c in s.cameras] if cfg.tracking_scale != 1.0 else cams
    train_idx = [i for i in range(len(cams)) if i not in exclude_cameras]
    frame0_labels = s.labels()
    for i, cam in enumerate(cams):
        cam.valid_ids = torch.as_tensor([bool((frame0_labels[i] == k).any()) for k in persons.ids], device=dev)
    tracker = Tracker(persons, topo, cfg)
    for i, cam in enumerate(cams):
        full = frame0_labels[i]
        cam.labels = torch.as_tensor(cv2.resize(full.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST).astype(np.int64) if cfg.tracking_scale != 1.0 else full, device=dev)
    anchor0, anchor0_ok = mask_centroid_anchor([cams[i] for i in train_idx], persons.ids, cfg.min_mask_area_px // 4, 3)
    states = [identity_state(len(topo.rest), dev)]; metrics = []
    vm_report = json.loads((s.out / masks / "report.json").read_text()) if (s.out / masks / "report.json").is_file() else {}
    mask_s = sum(v["propagate_s_per_frame"] for k, v in vm_report.items() if int(k) in train_idx)
    for f in range(start + 1, end):
        a = time.time()
        import cv2
        for i in train_idx:
            cam = cams[i]; img = frame_image(s, i, f); lab = load_video_labels(s.out / masks / f"cam_{i:02d}", f)
            if cfg.tracking_scale != 1.0:
                full_cams[i].labels = torch.as_tensor(lab, device=dev)
                img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
                lab = cv2.resize(lab.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            cam.image = torch.as_tensor(img, device=dev).float().div(255)
            cam.labels = torch.as_tensor(lab, device=dev)
            ys, xs = torch.nonzero(cam.labels > 0, as_tuple=True)
            mgn = cfg.tracking_box_margin_px
            cam.box = (max(int(xs.min()) - mgn, 0), max(int(ys.min()) - mgn, 0), min(int(xs.max()) + mgn + 1, cam.width), min(int(ys.max()) + mgn + 1, cam.height)) if len(xs) else (0, 0, cam.width, cam.height)
        io_s = time.time() - a
        prev = states[-1]; prev2 = states[-2] if len(states) > 1 else None
        init = warm_start(prev, prev2, cfg.tracking_warm_start)
        if cfg.tracking_centroid_anchor:
            # Re-anchor each person's bulk translation on the triangulated mask centroid (cheap,
            # multi-view, pose-independent up to articulation): lost persons are re-acquired.
            X, ok = mask_centroid_anchor([cams[i] for i in train_idx], persons.ids, cfg.min_mask_area_px // 4, 3)
            ok &= anchor0_ok
            shift = torch.zeros_like(X).index_reduce_(0, tracker.cp_inst_index, init.t, "mean", include_self=False)
            target = torch.where(ok[:, None], X - anchor0, shift)
            init.t = init.t + (target - shift)[tracker.cp_inst_index]
            anchor = (target, ok)
        else: anchor = None
        a = time.time()
        state, info = tracker.track_frame([cams[i] for i in train_idx], init, prev, prev2, anchor)
        info.update({"frame": f, "io_s": io_s, "track_s": time.time() - a, "mask_propagation_s": mask_s})
        states.append(state); metrics.append(info)
        if cfg.tracking_save_ply: export_deformed_ply(persons, tracker, state, d / "ply" / f"frame_{f:06d}.ply")
        print(f"  frame {f:4d}: {info['iterations']:3d} it, loss {info['first_loss']:.4f}->{info['final_loss']:.4f}, track {info['track_s']:.2f}s "
              f"(mls {info['mls_fwd']:.2f} render {info['render']:.2f} bwd {info['backward']:.2f} opt {info['optimizer']:.2f}) io {io_s:.2f}s")
        if (f - start) % cfg.eval_frame_stride == 0 or f == end - 1:
            ev = evaluate_frame(s, persons, tracker, state, models.get(0), f, full_cams, exclude_cameras)
            metrics[-1]["eval"] = ev; print(f"    eval frame {f}: {json.dumps({k: round(v, 3) for k, v in ev.items() if isinstance(v, float)})}")
        if f % 25 == 0 or f == end - 1: _save_states(d, states, topo, start, metrics, setup_s)
    _save_states(d, states, topo, start, metrics, setup_s)
    return summarize(d, metrics)


def _save_states(d: Path, states, topo, start: int, metrics: list, setup_s: float) -> None:
    import torch
    t = torch.stack([x.t for x in states]).cpu().numpy(); q = torch.stack([x.q for x in states]).cpu().numpy()
    np.savez_compressed(d / "cp_states.npz", t=t, q=q, frames=np.arange(start, start + len(states)), cp_instance=topo.cp_instance.cpu().numpy(), rest=topo.rest.cpu().numpy())
    (d / "metrics.json").write_text(json.dumps({"setup_s": setup_s, "frames": metrics}, indent=2, default=float) + "\n")


def summarize(d: Path, metrics: list) -> dict:
    keys = ("track_s", "mls_fwd", "render", "backward", "optimizer", "regularizers", "io_s", "mask_propagation_s", "iterations")
    summary = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
    ev = [m["eval"] for m in metrics if "eval" in m]
    if ev:
        keys = sorted({k for e in ev for k, v in e.items() if isinstance(v, float)})
        for k in keys: summary[f"eval_{k}"] = float(np.mean([e[k] for e in ev if k in e]))
        held = [e for e in ev if "heldout_psnr" in e]
        if held: summary["drift_heldout_psnr_first_last"] = [held[0]["heldout_psnr"], held[-1]["heldout_psnr"]]
        summary["drift_train_iou_first_last"] = [ev[0]["train_iou"], ev[-1]["train_iou"]]
    (d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))
    return summary


def evaluate_frame(s: Scene, persons, tracker, state, bg_model, f: int, cams, exclude_cameras) -> dict:
    """Held-out cameras (never used in tracking) for frame f + person IoU on training cameras."""
    import cv2, torch
    from ring_init.deform.track import render_persons
    from ring_init.eval.heldout import load_all_cameras
    from ring_init.gs.train import ssim
    cfg = s.cfg; dev = cfg.device; out = {}
    with torch.no_grad():
        means, quats, sh1 = tracker.deform(state.t, state.q)
        ious = []
        for i, cam in enumerate(cams):
            if cam.labels is None: continue
            _, _, alpha = render_persons(persons, means, quats, sh1, cam)
            a = alpha > 0.5; g = cam.labels > 0
            ious.append(float((a & g).sum() / max(int((a | g).sum()), 1)))
        out["train_iou"] = float(np.mean(ious))
        for c in exclude_cameras:  # leave-one-camera-out: a training camera never used in tracking
            cam = cams[c]; img = torch.as_tensor(frame_image(s, c, f), device=dev).float().div(255)
            rgb, _, alpha = render_persons(persons, means, quats, sh1, cam)
            comp = (rgb + (1 - alpha[..., None]) * cam.background).clamp(0, 1)
            out[f"loo_cam{c:02d}_psnr"] = float(-10 * torch.log10(((comp - img) ** 2).mean()))
        eval_dir = s.out / "eval_frames"
        if cfg.eval_sparse and eval_dir.is_dir():
            import lpips
            if not hasattr(evaluate_frame, "_lp"): evaluate_frame._lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
            if not hasattr(evaluate_frame, "_cams"): evaluate_frame._cams = {}
            allc = load_all_cameras(cfg.eval_sparse); train_views = {source_view(c) for c in s.cameras}
            views = sorted(v for v in allc if v not in train_views)[:cfg.tracking_eval_cameras]
            ps, ss, ls, pps = [], [], [], []
            for v in views:
                path = eval_dir / f"view_{v:03d}" / f"{f:06d}.png"
                if not path.is_file(): continue
                if v not in evaluate_frame._cams: evaluate_frame._cams[v] = _camera_frame(allc[v], bg_model, dev)
                cam = evaluate_frame._cams[v]
                img = torch.as_tensor(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), device=dev).float().div(255)
                rgb, _, alpha = render_persons(persons, means, quats, sh1, cam)
                comp = (rgb + (1 - alpha[..., None]) * cam.background).clamp(0, 1)
                ps.append(float(-10 * torch.log10(((comp - img) ** 2).mean())))
                roi = F_dilate(alpha > 0.05, 6)
                pps.append(float(-10 * torch.log10(((comp - img) ** 2)[roi].mean())))
                P, T = comp.permute(2, 0, 1), img.permute(2, 0, 1)
                ss.append(float(ssim(P, T))); ls.append(float(evaluate_frame._lp(P[None] * 2 - 1, T[None] * 2 - 1).mean()))
            if ps:
                out.update({"heldout_psnr": float(np.mean(ps)), "heldout_ssim": float(np.mean(ss)), "heldout_lpips": float(np.mean(ls)),
                            "heldout_person_psnr": float(np.mean(pps)), "heldout_views": len(ps)})
    return out


def F_dilate(mask, px: int):
    import torch.nn.functional as F
    return F.max_pool2d(mask.float()[None, None], 2 * px + 1, 1, px)[0, 0] > 0


def export_deformed_ply(persons, tracker, state, path: Path) -> None:
    """Deformed person Gaussians of one frame in the standard 3DGS layout (disk heavy: behind
    config.tracking_save_ply)."""
    import torch
    from ring_init.gs.export import save_ply
    from ring_init.gs.train import GaussianModel
    with torch.no_grad():
        means, quats, sh1 = tracker.deform(state.t, state.q)
        shN = torch.cat((sh1, torch.zeros_like(sh1[:, :0])), 1) if sh1 is not None else torch.zeros(len(means), 3, 3, device=means.device)
        model = GaussianModel({"means": means, "quats": quats, "scales": persons.scales.log(), "opacities": torch.logit(persons.opacities.clamp(1e-6, 1 - 1e-6)),
                               "sh0": persons.sh0, "shN": shN}, persons.instance_ids)
    path.parent.mkdir(parents=True, exist_ok=True); save_ply(model, path)


# --------------------------------------------------------------------------- Stage B v2 (whole scene)
class _VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int = 25):
        import subprocess
        path.parent.mkdir(parents=True, exist_ok=True); self.size = (width, height)
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)], stdin=subprocess.PIPE)

    def write_bytes(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())

    def write(self, rgb) -> None:
        import torch
        img = torch.nn.functional.interpolate(rgb.permute(2, 0, 1)[None], size=(self.size[1], self.size[0]), mode="area")[0].permute(1, 2, 0)
        self.proc.stdin.write((img.clamp(0, 1).cpu().numpy() * 255 + .5).astype(np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close(); self.proc.wait()


def _decode_video(path: Path, count: int, width: int, height: int):
    """First `count` RGB frames of an encoded video, which must be `width` x `height`."""
    import subprocess
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        for i in range(count):
            buf = proc.stdout.read(size)
            if len(buf) < size: raise RuntimeError(f"{path}: only {i} of {count} frames at {width}x{height}")
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close(); proc.wait()


def _full_view(cam, img, device: str):
    import torch
    from ring_init.gs.train import View
    w2c = np.eye(4, dtype=np.float32); w2c[:3] = np.column_stack((cam.R, cam.t))
    ones = torch.ones((cam.height, cam.width), dtype=torch.bool, device=device)
    return View(image=torch.as_tensor(img, device=device).permute(2, 0, 1).float() / 255, loss_mask=ones, alpha_target=ones.float(), alpha_ignore=~ones,
                viewmat=torch.as_tensor(w2c, device=device), K=torch.as_tensor(cam.K, dtype=torch.float32, device=device), width=cam.width, height=cam.height, name=cam.name)


def step_track_scene(s: Scene, start: int, end: int, tag: str = "stage_b_v2", resume_from: tuple[str, int] | None = None) -> dict:
    """QuickCapture-style: every Gaussian deforms (controls everywhere, denser on players), with
    keyframe retraining. Frame `start` = Stage A scene."""
    import cv2, torch
    from ring_init.deform.control_points import build_scene_controls
    from ring_init.deform.scene_track import SceneTracker
    from ring_init.deform.track import mask_centroid_anchor
    from ring_init.eval.heldout import load_all_cameras
    from ring_init.gs.export import concat_models, save_ply
    from ring_init.masks.video import load_video_labels
    from ring_init.render_video import orbit_cameras
    cfg = scaled(s.cfg, s); dev = cfg.device; d = s.dir(tag); cfg.save(d / "config.json")
    from ring_init.ball import BALL_ID
    use_ball = cfg.ball and (s.out / "canonical_ball" / "manifest.json").is_file()
    models = load_canonical(s, "canonical_ball" if use_ball else "canonical"); model = concat_models([models[k] for k in sorted(models)])
    person_ids = sorted(k for k in models if k != 0)          # movable groups (players, referees, staff, ball)
    t0 = time.time()
    ctrl = build_scene_controls(model.params["means"].detach(), model.params["instance_ids"].long(), {0: cfg.control_points_background, BALL_ID: cfg.ball_control_points, -1: cfg.control_points},
                                cfg.gaussian_control_knn, cfg.control_knn, cfg.mls_sigma_knn)
    ball_traj = np.load(s.out / "ball" / "trajectory.npz") if use_ball else None
    bi = person_ids.index(BALL_ID) if use_ball else None
    anchor_w = torch.full((len(person_ids),), cfg.tracking_anchor_weight, device=dev)
    if use_ball: anchor_w[bi] = cfg.ball_anchor_weight
    setup_s = time.time() - t0; print(f"  controls: {len(ctrl.pos)} ({int((ctrl.group == 0).sum())} background) in {setup_s:.1f}s; gaussians {len(model)}")
    tracker = SceneTracker(model, ctrl, person_ids, cfg); tracker.remember_canonical()
    cams = [_camera_frame(c, None, dev, cfg.tracking_scale) for c in s.cameras]
    frame0 = s.labels()
    for i, cam in enumerate(cams):
        cam.valid_ids = torch.as_tensor([bool((frame0[i] == k).any()) for k in person_ids], device=dev)
        cam.labels = torch.as_tensor(cv2.resize(frame0[i].astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST).astype(np.int64), device=dev)
    anchor0, anchor0_ok = mask_centroid_anchor(cams, person_ids, cfg.min_mask_area_px // 4, 3)
    allc = load_all_cameras(cfg.eval_sparse); train_views = {source_view(c) for c in s.cameras}
    heldout = sorted(v for v in allc if v not in train_views)[:cfg.tracking_eval_cameras]
    vid_cams = {v: _camera_frame(allc[v], None, dev, cfg.v2_video_scale) for v in cfg.v2_video_views}
    orbit = orbit_cameras(s.cameras, end - start, 360.0)
    writers = {v: _VideoWriter(d / "videos" / f"heldout_view{v:02d}.mp4", c.width // 2 * 2, c.height // 2 * 2) for v, c in vid_cams.items()}
    oc0 = _camera_frame(orbit[0], None, dev, cfg.v2_video_scale); writers["orbit"] = _VideoWriter(d / "videos" / "orbit360.mp4", oc0.width // 2 * 2, oc0.height // 2 * 2)
    positions = [ctrl.pos.cpu().numpy()]; metrics = []; refiner = None; repair = None; boost = torch.ones(len(person_ids), device=dev)
    frame0_ids = [int(k) for k in np.unique(np.concatenate([np.unique(l) for l in frame0])) if k > 0]
    resume_frame = None
    if resume_from is not None:
        # Continue from another run's full checkpoint (same canonical scene and controls). Frame
        # numbering, mask/keyframe cadence and the orbit path stay tied to `start`, so the resumed
        # run schedules exactly the same frames as an uninterrupted one.
        src = s.out / resume_from[0]; resume_frame = int(resume_from[1])
        st = torch.load(src / "state" / f"frame_{resume_frame:06d}.pt", map_location="cpu", weights_only=False)
        if st["frame"] != resume_frame or st["start"] != start: raise ValueError(f"checkpoint {st['frame']}/{st['start']} does not match resume {resume_frame}/{start}")
        tracker.restore_state(st["tracker"]); boost = st["boost"].to(dev)
        if st["repair"] is not None:
            from ring_init.masks.reassociate import IdentityRepair
            repair = IdentityRepair(s, cfg, [k for k in person_ids if k in set(frame0_ids)])
            repair.ref_pos, repair.ref_hist = st["repair"]["ref_pos"], st["repair"]["ref_hist"]
        positions = list(np.load(src / "control_positions.npz")["pos"][: resume_frame - start + 1])
        metrics = [m for m in json.loads((src / "metrics.json").read_text())["frames"] if m["frame"] <= resume_frame]
        if len(positions) != resume_frame - start + 1: raise ValueError("source control_positions.npz ends before the checkpoint")
        (d / "resumed_from.json").write_text(json.dumps({"source": resume_from[0], "frame": resume_frame}) + "\n")
        print(f"  resumed from {src.name} frame {resume_frame} ({len(tracker.model)} Gaussians)")

    def save_state(f: int) -> None:
        torch.save({"frame": f, "start": start, "tracker": tracker.checkpoint_state(), "boost": boost.cpu(),
                    "repair": None if repair is None else {"ref_pos": repair.ref_pos, "ref_hist": repair.ref_hist}},
                   s.dir(f"{tag}/state") / f"frame_{f:06d}.pt")

    def write_videos(f: int) -> None:
        with torch.no_grad():
            means, quats, _, _, _, sh1 = tracker.activated()
            for v, cam in vid_cams.items(): writers[v].write(tracker.render(means, quats, sh1, cam, False)[0])
            oc = _camera_frame(orbit[f - start], None, dev, cfg.v2_video_scale); writers["orbit"].write(tracker.render(means, quats, sh1, oc, False)[0])

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1); loader: dict = {}

    def load_frame(fr: int):
        """CPU side of one frame: full-res images, tracking-res images, and (full-detail mode) labels."""
        full = [frame_image(s, i, fr) for i in range(len(cams))]
        small = [cv2.resize(im, (c.width, c.height), interpolation=cv2.INTER_AREA) for im, c in zip(full, cams)]
        labs = None
        if cfg.v2_mask_every <= 0:
            labs = [cv2.resize(load_video_labels(s.out / "video_masks" / f"cam_{i:02d}", fr).astype(np.uint8), (c.width, c.height), interpolation=cv2.INTER_NEAREST).astype(np.int64) for i, c in enumerate(cams)]
        return full, small, labs

    if cfg.v2_dynamic_only_deform: tracker.cache_background(cams)
    if resume_frame is None: write_videos(start)
    else:   # copy the source run's frames start..resume_frame so the videos cover the whole range
        for key, w in writers.items():
            name = "orbit360.mp4" if key == "orbit" else f"heldout_view{key:02d}.mp4"
            for rgb in _decode_video(s.out / resume_from[0] / "videos" / name, resume_frame - start + 1, *w.size): w.write_bytes(rgb)
    for f in range((start if resume_frame is None else resume_frame) + 1, end):
        a = time.time(); mask_frame = cfg.v2_mask_every <= 0 or (f - start) % cfg.v2_mask_every == 0
        full_imgs, small, labs = (loader.pop(f).result() if f in loader else load_frame(f))
        if cfg.v2_prefetch and f + 1 < end: loader[f + 1] = pool.submit(load_frame, f + 1)   # overlaps with GPU work
        for i, cam in enumerate(cams):
            cam.image = torch.as_tensor(small[i], device=dev).float().div(255)
            if labs is not None: cam.labels = torch.as_tensor(labs[i], device=dev)
        io_s = time.time() - a; a = time.time()
        if cfg.v2_mask_every > 0 and cfg.v2_mask_source == "reassoc":
            if mask_frame:   # SAM2 video segments with identities rebuilt in 3D (view-consistent)
                if repair is None:
                    from ring_init.masks.reassociate import IdentityRepair
                    repair = IdentityRepair(s, cfg, [k for k in person_ids if k in set(frame0_ids)])
                raw = [load_video_labels(s.out / "video_masks" / f"cam_{i:02d}", f) for i in range(len(cams))]
                fixed, rinfo = repair(raw, full_imgs); info_pre = {"reassoc": rinfo}
                for cam, lab in zip(cams, fixed):
                    cam.labels = torch.as_tensor(cv2.resize(lab.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST).astype(np.int64), device=dev)
            else:
                for cam in cams: cam.labels = None
        elif cfg.v2_mask_every > 0 and cfg.v2_mask_source == "video":
            for i, cam in enumerate(cams):   # SAM2 video labels, used only on mask frames
                cam.labels = torch.as_tensor(cv2.resize(load_video_labels(s.out / "video_masks" / f"cam_{i:02d}", f).astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST).astype(np.int64), device=dev) if mask_frame else None
        elif cfg.v2_mask_every > 0:
            if mask_frame:   # keyframe masks prompted from the tracked people
                if refiner is None:
                    from ring_init.masks.segment import SAM2Refiner
                    refiner = SAM2Refiner(cfg.sam2_checkpoint, cfg.sam2_config, dev)
                for cam, lab in zip(cams, tracker.reprompt_labels(cams, full_imgs, refiner)): cam.labels = lab
            else:
                for cam in cams: cam.labels = None
        masks_s = time.time() - a
        if mask_frame:
            X, ok = mask_centroid_anchor(cams, person_ids, cfg.min_mask_area_px // 4, 3)
        else:
            X, ok = torch.zeros(len(person_ids), 3, device=dev), torch.zeros(len(person_ids), dtype=torch.bool, device=dev)
        disp, valid = X - anchor0, (ok & anchor0_ok) if cfg.tracking_centroid_anchor else torch.zeros_like(ok)
        if use_ball:   # the ball follows its triangulated trajectory (no SAM2 mask for it)
            p = ball_traj["pos"]; n = f - int(ball_traj["frames"][0])
            if n < len(p) and np.isfinite(p[n]).all():
                disp[bi] = torch.as_tensor(p[n] - p[0], dtype=disp.dtype, device=dev); valid[bi] = True
            else: valid[bi] = False
        anchor = (disp, valid, anchor_w * boost)
        a = time.time()
        info = tracker.deform_frame(cams, anchor, use_masks=mask_frame); info["deform_s"] = time.time() - a; info["masks_s"] = masks_s
        if cfg.v2_mask_source == "reassoc" and mask_frame: info.update(info_pre)
        if (f - start) % cfg.v2_keyframe_every == 0:
            if cfg.v2_health_iou > 0:   # lost / exploded people: re-initialize from canonical at their anchor position
                for cam, i in zip(cams, range(len(cams))):
                    if cam.labels is None:
                        cam.labels = torch.as_tensor(cv2.resize(load_video_labels(s.out / "video_masks" / f"cam_{i:02d}", f).astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST).astype(np.int64), device=dev)
                iou = tracker.person_iou(cams); info["health_iou"] = {int(k): round(float(v), 3) for k, v in zip(person_ids, iou)}
                reinit = []
                for j, k in enumerate(person_ids):
                    if bool(valid[j]) and float(iou[j]) < cfg.v2_health_iou:
                        tracker.reinit_person(k, disp[j] + (tracker.cp_pos0_mean[j] - tracker.canon_ctrl[tracker.ctrl.group == k].mean(0))); reinit.append(k)
                boost = torch.where(torch.nan_to_num(iou, nan=1.0) < cfg.v2_boost_iou, cfg.v2_anchor_boost, 1.0).to(dev)
                info["reinit"] = reinit
                if reinit: print(f"    re-initialized {reinit} (IoU {[info['health_iou'][k] for k in reinit]})")
            views = [_full_view(c, full_imgs[i], dev) for i, c in enumerate(s.cameras)]
            focus = cfg.v2_keyframe_focus and (f - start) % cfg.v2_keyframe_full_every != 0
            info.update(tracker.keyframe(views, d / "keyframes" / f"log_{f:06d}.jsonl", focus)); del views; torch.cuda.empty_cache()
            if cfg.v2_dynamic_only_deform: tracker.cache_background(cams)
        elif cfg.v2_refine_iterations > 0:
            # Full-model inter-keyframe refinement is the direct sawtooth experiment.
            # It never densifies, avoiding a new control binding/model-size discontinuity
            # at every video frame; the scheduled keyframes retain that responsibility.
            views = [_full_view(c, full_imgs[i], dev) for i, c in enumerate(s.cameras)]
            r = tracker.refine(views, d / "refinements" / f"log_{f:06d}.jsonl",
                               cfg.v2_refine_iterations, cfg.v2_refine_views_per_step,
                               cfg.v2_refine_focus, dynamic_only=cfg.v2_refine_dynamic_only and cfg.v2_refine_focus)
            info.update({"refine_s": r["refine_s"], "refine_final_loss": r["refine_final_loss"]})
            del views; torch.cuda.empty_cache()
            if cfg.v2_dynamic_only_deform: tracker.cache_background(cams)
        else:
            info["touchup_s"] = tracker.touch_up(cams)
        info.update({"frame": f, "io_s": io_s, "gaussians": len(tracker.model)})
        positions.append(tracker.ctrl.pos.cpu().numpy())
        write_videos(f)
        print(f"  frame {f:4d}: {info['iterations']:3d} it, loss {info['first_loss']:.4f}->{info['final_loss']:.4f}, deform {info['deform_s']:.2f}s"
              + (f", keyframe {info['keyframe_s']:.1f}s ({info['gaussians_before']}->{info['gaussians_after']})" if "keyframe_s" in info
                 else f", refine {info['refine_s']:.1f}s" if "refine_s" in info else f", touch-up {info.get('touchup_s', 0):.2f}s"))
        if (f - start) % cfg.eval_frame_stride == 0 or f == end - 1:
            info["eval"] = evaluate_scene_frame(s, tracker, f, heldout, allc, dev)
            print(f"    eval frame {f}: {json.dumps({k: round(v, 3) for k, v in info['eval'].items()})}")
        metrics.append(info)
        if (f - start) % cfg.v2_save_ply_every == 0: save_ply(tracker.model, d / "ply" / f"frame_{f:06d}" / "point_cloud.ply")
        save_now = cfg.v2_save_state_every > 0 and (f - start) % cfg.v2_save_state_every == 0
        if f % 25 == 0 or f == end - 1 or save_now:
            np.savez_compressed(d / "control_positions.npz", pos=np.stack(positions), group=ctrl.group.cpu().numpy(), frames=np.arange(start, start + len(positions)))
            (d / "metrics.json").write_text(json.dumps({"setup_s": setup_s, "frames": metrics}, indent=2, default=float) + "\n")
        if save_now: save_state(f)
    pool.shutdown(wait=False)
    for w in writers.values(): w.close()
    save_ply(tracker.model, d / "ply" / f"frame_{end - 1:06d}" / "point_cloud.ply")
    return summarize_scene(d, metrics)


def evaluate_scene_frame(s: Scene, tracker, f: int, heldout: list[int], allc: dict, dev: str) -> dict:
    """Held-out (never used) + training-view PSNR, each full frame and person region (rendered
    person weight > 0.05, dilated 6 px); training IoU of rendered person weight vs SAM2 masks."""
    import cv2, lpips, torch
    from ring_init.gs.train import ssim
    from ring_init.masks.video import load_video_labels
    if not hasattr(evaluate_scene_frame, "_lp"): evaluate_scene_frame._lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    out = {}
    with torch.no_grad():
        means, quats, _, _, _, sh1 = tracker.activated()
        def score(cam, img):
            rgb, inst, _ = tracker.render(means, quats, sh1, cam, True); rgb = rgb.clamp(0, 1); pw = inst.sum(-1)
            roi = F_dilate(pw > 0.05, 6); err = (rgb - img) ** 2
            return rgb, pw, float(-10 * torch.log10(err.mean())), float(-10 * torch.log10(err[roi].mean())) if bool(roi.any()) else float("nan")
        tr, trp, ious = [], [], []
        for i, c in enumerate(s.cameras):
            cam = _camera_frame(c, None, dev); img = torch.as_tensor(frame_image(s, i, f), device=dev).float().div(255)
            _, pw, p, pp = score(cam, img); tr.append(p); trp.append(pp)
            g = torch.as_tensor(load_video_labels(s.out / "video_masks" / f"cam_{i:02d}", f), device=dev) > 0; a = pw > 0.5
            ious.append(float((a & g).sum() / max(int((a | g).sum()), 1)))
        out.update({"train_psnr": float(np.mean(tr)), "train_person_psnr": float(np.nanmean(trp)), "train_iou": float(np.mean(ious))})
        hp, hpp, hs, hl = [], [], [], []
        for v in heldout:
            path = s.out / "eval_frames" / f"view_{v:03d}" / f"{f:06d}.png"
            if not path.is_file(): continue
            cam = _camera_frame(allc[v], None, dev); img = torch.as_tensor(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), device=dev).float().div(255)
            rgb, _, p, pp = score(cam, img); hp.append(p); hpp.append(pp)
            P, T = rgb.permute(2, 0, 1), img.permute(2, 0, 1); hs.append(float(ssim(P, T))); hl.append(float(evaluate_scene_frame._lp(P[None] * 2 - 1, T[None] * 2 - 1).mean()))
        if hp: out.update({"heldout_psnr": float(np.mean(hp)), "heldout_person_psnr": float(np.nanmean(hpp)), "heldout_ssim": float(np.mean(hs)), "heldout_lpips": float(np.mean(hl))})
    return out


def summarize_scene(d: Path, metrics: list) -> dict:
    keys = ("deform_s", "mls_fwd", "render", "backward", "optimizer", "regularizers", "io_s", "iterations")
    summary = {k: float(np.mean([m[k] for m in metrics if k in m])) for k in keys}
    kf = [m["keyframe_s"] for m in metrics if "keyframe_s" in m]; rf = [m["refine_s"] for m in metrics if "refine_s" in m]; tu = [m["touchup_s"] for m in metrics if "touchup_s" in m]
    summary.update({"keyframe_s": float(np.mean(kf)) if kf else 0.0, "refine_s": float(np.mean(rf)) if rf else 0.0, "touchup_s": float(np.mean(tu)) if tu else 0.0,
                    "seconds_per_frame": float(np.mean([m["deform_s"] + m.get("keyframe_s", 0) + m.get("refine_s", 0) + m.get("touchup_s", 0) + m["io_s"] + m.get("masks_s", 0) for m in metrics])),
                    "masks_s": float(np.mean([m.get("masks_s", 0) for m in metrics])),
                    "final_gaussians": metrics[-1]["gaussians"]})
    ev = [m["eval"] for m in metrics if "eval" in m]
    for k in sorted({k for e in ev for k in e}):
        summary[f"eval_{k}"] = float(np.nanmean([e[k] for e in ev if k in e]))
    held = [e for e in ev if "heldout_psnr" in e]
    if held: summary["drift_heldout"] = {k: [held[0][k], held[-1][k]] for k in ("heldout_psnr", "heldout_person_psnr")}
    summary["drift_train"] = {k: [ev[0][k], ev[-1][k]] for k in ("train_psnr", "train_person_psnr", "train_iou")}
    (d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))
    return summary
