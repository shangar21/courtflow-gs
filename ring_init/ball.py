"""The ball: its own instance (id BALL_ID) with its own control points.

Frame 0: detect the ball ("sports ball") in the training views, triangulate centre and radius with
the known cameras (RANSAC over view pairs), build a sphere of Gaussians coloured from the views, and
remove any person/background Gaussians inside it (it was absorbed into the dribbler / background).
Frames 1..: causal tracking — gravity-aware constant-velocity prediction, detection in crops around
the predicted projection, RANSAC triangulation, floor-bounce prior when unobserved. The trajectory
anchors the ball's control points in Stage B."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from ring_init.config import Config
from ring_init.io.calib import Camera

BALL_ID = 99
COCO_SPORTS_BALL = 37


class BallDetector:
    def __init__(self, cfg: Config):
        from torchvision.models import detection
        self.model = detection.fasterrcnn_resnet50_fpn_v2(weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=cfg.detector_max_detections).to(cfg.device).eval()
        self.cfg = cfg

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, box: tuple[int, int, int, int] | None = None) -> list[tuple[np.ndarray, float, float]]:
        """Returns [(centre uv, radius px, score)] in full-image pixels (optionally inside a crop)."""
        c = self.cfg; x0, y0 = 0, 0
        if box is not None:
            x0, y0, x1, y1 = box; rgb = rgb[y0:y1, x0:x1]
            self.model.transform.min_size = (c.ball_crop_detector_size,); self.model.transform.max_size = c.ball_crop_detector_size * 2
        else:
            self.model.transform.min_size = (c.detector_min_size,); self.model.transform.max_size = c.detector_max_size
        out = self.model([torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).to(c.device)])[0]
        keep = (out["labels"] == COCO_SPORTS_BALL) & (out["scores"] >= c.ball_score_threshold)
        res = []
        for b, sc in zip(out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()):
            res.append((np.array([(b[0] + b[2]) / 2 + x0, (b[1] + b[3]) / 2 + y0]), float(max(b[2] - b[0], b[3] - b[1]) / 2), float(sc)))
        return res


def _ray(cam: Camera, uv: np.ndarray) -> np.ndarray:
    d = cam.R.T @ np.linalg.inv(cam.K) @ np.r_[uv, 1.0]; return d / np.linalg.norm(d)


def _intersect(cams: list[Camera], uvs: list[np.ndarray], w: list[float]) -> np.ndarray:
    A = np.zeros((3, 3)); b = np.zeros(3)
    for cam, uv, wi in zip(cams, uvs, w):
        d = _ray(cam, uv); P = np.eye(3) - np.outer(d, d); A += wi * P; b += wi * P @ cam.center
    return np.linalg.solve(A + 1e-12 * np.eye(3), b)


def triangulate(obs: list[tuple[int, np.ndarray, float, float]], cameras: list[Camera], reproj_px: float) -> tuple[np.ndarray | None, list[int]]:
    """obs = [(camera index, uv, radius px, score)], at most one per camera. RANSAC over pairs,
    then a score-weighted least-squares refit on the inliers."""
    best, best_in = None, []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            ci, cj = obs[i][0], obs[j][0]
            X = _intersect([cameras[ci], cameras[cj]], [obs[i][1], obs[j][1]], [1, 1])
            inl = []
            for k, (c, uv, _, _) in enumerate(obs):
                p, z = cameras[c].project(X[None])
                if z[0] > 0 and np.linalg.norm(p[0] - uv) < reproj_px: inl.append(k)
            if len(inl) > len(best_in): best, best_in = X, inl
    if best is None or len(best_in) < 2: return None, []
    X = _intersect([cameras[obs[k][0]] for k in best_in], [obs[k][1] for k in best_in], [obs[k][3] for k in best_in])
    return X, [obs[k][0] for k in best_in]


def detect_all(detector: BallDetector, images: list[np.ndarray], cameras: list[Camera], predicted: np.ndarray | None, crop: int) -> list:
    """One (best) detection per camera; crops around the predicted projection when available."""
    obs = []
    for ci, (img, cam) in enumerate(zip(images, cameras)):
        box = None
        if predicted is not None:
            p, z = cam.project(predicted[None])
            if z[0] <= 0: continue
            u, v = p[0]; h = crop // 2
            x0, y0 = int(np.clip(u - h, 0, cam.width - crop)), int(np.clip(v - h, 0, cam.height - crop)); box = (x0, y0, x0 + crop, y0 + crop)
        dets = detector(img, box)
        if dets:
            uv, r, sc = max(dets, key=lambda d: d[2]); obs.append((ci, uv, r, sc))
    return obs


def ball_radius(X: np.ndarray, obs: list, cameras: list[Camera], inliers: list[int], nominal: float) -> float:
    rs = []
    for c, uv, r_px, _ in obs:
        if c in inliers:
            _, z = cameras[c].project(X[None]); rs.append(r_px * z[0] / cameras[c].K[0, 0])
    return float(np.clip(np.median(rs), 0.7 * nominal, 1.3 * nominal)) if rs else nominal


def track_ball(s, start: int, end: int) -> dict:
    """Causal ball trajectory for frames start..end-1 (training cameras only). Writes ball/trajectory.npz."""
    from ring_init.stage_a import scaled
    from ring_init.stage_b import frame_image
    cfg = scaled(s.cfg, s); cams = s.cameras; d = s.dir("ball"); det = BallDetector(cfg)
    dt = 1.0 / cfg.video_fps; g = -9.81 * cfg.units_per_meter * s.floor().normal     # gravity along -up
    floor = s.floor(); nominal = cfg.m(cfg.ball_radius_m)
    pos = np.full((end - start, 3), np.nan); observed = np.zeros(end - start, bool); nviews = np.zeros(end - start, int)
    x = v = None; radius = nominal; lost = 0; t0 = time.time(); early = []
    for n, f in enumerate(range(start, end)):
        images = [frame_image(s, i, f) for i in range(len(cams))]
        pred = None if x is None else x + v * dt + 0.5 * g * dt * dt
        crop = int(cfg.ball_crop_px * (1 + lost))  # widen the search while the ball is unobserved
        obs = detect_all(det, images, cams, pred if (pred is not None and lost < cfg.ball_max_lost) else None, min(crop, min(c.height for c in cams)))
        X, inl = triangulate(obs, cams, cfg.ball_reproj_px) if len(obs) >= 2 else (None, [])
        reacquire = x is None or lost >= cfg.ball_max_lost        # (re-)initialization: no motion gate, but >= 3 agreeing views
        if X is not None and not reacquire and np.linalg.norm(X - pred) > cfg.m(cfg.ball_max_jump_m): X = None   # implausible jump
        if X is not None and reacquire and len(inl) < cfg.ball_init_min_views: X = None   # static look-alikes (rim) agree in 2 views
        if x is None: early.append((n, obs))
        if X is not None:
            if x is None: radius = ball_radius(X, obs, cams, inl, nominal)
            v = (X - x) / dt if (x is not None and lost == 0) else np.zeros(3); x = X; lost = 0; observed[n] = True; nviews[n] = len(inl)
        elif x is not None:
            x, v = pred, (v + g * dt) * cfg.ball_lost_damping; lost += 1
            h = floor.height(x) - radius
            if h < 0:  # floor bounce prior
                vn = v @ floor.normal
                if vn < 0: v = v - (1 + cfg.ball_restitution) * vn * floor.normal
                x = x - h * floor.normal
        if x is not None: pos[n] = x
        if n % 25 == 0: print(f"  ball frame {f}: {'obs %d views' % nviews[n] if observed[n] else 'predicted' if x is not None else 'none'}")
    # Frames before the first reliable triangulation: single-view detections, closest point on the ray
    # to the first reliable position (temporal prior); otherwise hold that position.
    if observed.any():
        first = int(np.flatnonzero(observed)[0]); X0 = pos[first]
        for n, obs in early:
            if n >= first: continue
            cands = []
            for c, uv, _, sc in obs:
                cam = cams[c]; d_ = _ray(cam, uv); t_ = max((X0 - cam.center) @ d_, 0.0); P = cam.center + t_ * d_
                if np.linalg.norm(P - X0) < cfg.m(cfg.ball_max_jump_m) * (first - n): cands.append((np.linalg.norm(P - X0), P))
            pos[n] = min(cands, key=lambda c_: c_[0])[1] if cands else X0
    np.savez_compressed(d / "trajectory.npz", pos=pos, observed=observed, views=nviews, frames=np.arange(start, end), radius=radius)
    info = {"frames": end - start, "observed_fraction": float(observed.mean()), "radius_m": radius / cfg.units_per_meter, "seconds": time.time() - t0}
    (d / "trajectory.json").write_text(json.dumps(info, indent=2) + "\n"); print(f"[ball] {info}")
    return info


def build_ball_model(s, center: np.ndarray, radius: float, frame: int = 0):
    """Sphere of Gaussians coloured by the median colour over views that see each point."""
    from ring_init.gs.init_gaussians import initialize_gaussians
    from ring_init.stage_b import frame_image
    cfg = s.cfg; n = cfg.ball_gaussians
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i   # Fibonacci sphere
    normals = np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), 1); pts = center + radius * normals
    cols = []
    for ci, cam in enumerate(s.cameras):
        img = frame_image(s, ci, frame); uv, z = cam.project(pts)
        facing = ((cam.center - pts) * normals).sum(1) > 0
        u = np.clip(np.rint(uv[:, 0]).astype(int), 0, cam.width - 1); v = np.clip(np.rint(uv[:, 1]).astype(int), 0, cam.height - 1)
        c = np.where((facing & (z > 0))[:, None], img[v, u] / 255.0, np.nan); cols.append(c)
    with np.errstate(all="ignore"): col = np.nanmedian(np.stack(cols, 1), 1)
    col = np.where(np.isnan(col), np.nanmedian(col, 0), col)
    return initialize_gaussians(pts, col, normals, BALL_ID, cfg)


def step_ball_canonical(s) -> Path:
    """canonical_ball/: Stage A canonical + ball instance, with Gaussians inside the ball removed from
    the other instances. The Stage A canonical/ directory is left untouched."""
    import shutil
    from ring_init.gs.export import load_ply, save_ply
    from ring_init.gs.train import GaussianModel
    from ring_init.stage_a import scaled
    cfg = scaled(s.cfg, s); d = s.out / "canonical_ball"
    traj = np.load(s.out / "ball" / "trajectory.npz")
    if not traj["observed"].any(): raise RuntimeError("Ball never triangulated; cannot build the ball instance.")
    center, radius = traj["pos"][0], float(traj["radius"])   # frame 0 (backfilled if not triangulated there)
    manifest = json.loads((s.out / "canonical" / "manifest.json").read_text()); entries = []; removed = {}
    for e in manifest["instances"]:
        m = load_ply(e["model"], cfg.device); keep = (m.params["means"] - torch.as_tensor(center, dtype=torch.float32, device=cfg.device)).norm(dim=-1) > cfg.ball_clear_factor * radius
        removed[e["instance_id"]] = int((~keep).sum())
        m = GaussianModel({k: v[keep] for k, v in m.params.items() if k != "instance_ids"}, m.params["instance_ids"][keep])
        out = d / f"instance_{e['instance_id']:03d}" / "point_cloud.ply"; save_ply(m, out); entries.append({**e, "model": str(out), "gaussians": len(m)})
    ball = build_ball_model(s, center, radius, 0); out = d / f"instance_{BALL_ID:03d}" / "point_cloud.ply"; save_ply(ball, out)
    entries.append({"instance_id": BALL_ID, "kind": "ball", "model": str(out), "gaussians": len(ball), "center": center.tolist(), "radius": radius})
    (d / "manifest.json").write_text(json.dumps({**manifest, "instances": entries, "ball_removed_from": removed}, indent=2) + "\n")
    print(f"[ball canonical] centre {np.round(center, 4)} radius {radius / cfg.units_per_meter:.3f} m; removed {removed}")
    return d
