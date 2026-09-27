"""Stage A (frame 0) geometry steps. Each step caches its outputs under out/<scene>/<step>/ and
is skipped when they exist (delete the directory or pass --force <step> to recompute)."""
from __future__ import annotations
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import cv2
import numpy as np
import torch
from ring_init.config import Config
from ring_init.geom.floor import Floor, estimate_units_per_meter, fit_floor
from ring_init.geom.hull import inside_hull, instance_hull
from ring_init.geom.triangulate import filter_pair, triangulate_dlt
from ring_init.io.calib import Camera, load_json, save_colmap_text, undistort
from ring_init.io.crop import from_crop, square_box, to_crop

STEPS = ("calib", "match_full", "floor", "persons", "instances", "hulls", "match_crops", "fuse")


class Scene:
    """Paths, cameras and images shared by all steps."""

    def __init__(self, scene: str, cfg: Config, force: set[str] | None = None):
        self.cfg = cfg; self.name = scene
        self.source = Path(cfg.data_root) / scene; self.out = Path(cfg.out_root) / scene
        if not self.source.is_dir(): raise FileNotFoundError(f"Missing scene directory: {self.source}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.force = force or set()
        self._images: list[np.ndarray] | None = None
        self.timings_path = self.out / "timings.json"

    def dir(self, step: str) -> Path:
        d = self.out / step; d.mkdir(parents=True, exist_ok=True); return d

    def done(self, step: str, marker: str = "done.json") -> bool:
        return step not in self.force and (self.out / step / marker).is_file()

    def finish(self, step: str, payload: dict, seconds: float) -> None:
        payload = {**payload, "seconds": seconds}
        (self.dir(step) / "done.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
        timings = json.loads(self.timings_path.read_text()) if self.timings_path.is_file() else {}
        timings[step] = seconds; self.timings_path.write_text(json.dumps(timings, indent=2) + "\n")
        print(f"[{step}] {seconds:.1f}s {json.dumps({k: v for k, v in payload.items() if not isinstance(v, (list, dict))}, default=float)}")

    @property
    def cameras(self) -> list[Camera]:
        path = self.out / "calib" / "calibration_undistorted.json"
        if not path.is_file(): raise FileNotFoundError("Run the calib step first.")
        return load_json(path, self.cfg.camera_count)

    @property
    def images(self) -> list[np.ndarray]:
        """Undistorted RGB uint8 training images."""
        if self._images is None:
            self._images = [cv2.cvtColor(cv2.imread(str(self.out / "calib" / f"cam_{i:02d}.png")), cv2.COLOR_BGR2RGB) for i in range(self.cfg.camera_count)]
        return self._images

    def floor(self) -> Floor:
        return Floor.load(self.out / "floor" / "floor.json")

    def labels(self) -> list[np.ndarray]:
        from ring_init.masks.segment import load_label_masks
        cams = self.cameras
        return load_label_masks(self.out / "instances" / "masks", [(c.height, c.width) for c in cams])

    def instance_info(self) -> dict:
        return json.loads((self.out / "instances" / "instances.json").read_text())


# --------------------------------------------------------------------------- Step 1: calibration
def step_calib(s: Scene) -> None:
    if s.done("calib"): return
    t = time.time(); cfg = s.cfg; d = s.dir("calib")
    cameras = load_json(s.source / cfg.calibration, cfg.camera_count)
    updated = []
    for i, cam in enumerate(cameras):
        path = s.source / cfg.image_pattern.format(cam=i); image = cv2.imread(str(path))
        if image is None: raise FileNotFoundError(f"Missing synchronized first-frame image: {path}")
        if image.shape[:2] != (cam.height, cam.width): raise ValueError(f"{path}: image {image.shape[1]}x{image.shape[0]} != calibration {cam.width}x{cam.height}")
        image, new = undistort(image, cam); updated.append(new)
        cv2.imwrite(str(d / f"cam_{i:02d}.png"), image)
    payload = {"units": cfg.units, "cameras": [{"name": c.name, "K": c.K.tolist(), "distortion": [], "R": c.R.tolist(), "t": c.t.tolist(), "width": c.width, "height": c.height} for c in updated]}
    (d / "calibration_undistorted.json").write_text(json.dumps(payload, indent=2) + "\n")
    save_colmap_text(updated, d / "colmap")
    alignment = frusta_figure(updated, d / "calibration_frusta.png")
    s.finish("calib", {"alignment_cosine": alignment}, time.time() - t)


def frusta_figure(cameras: list[Camera], path: Path) -> list[float]:
    """Top/side views of the ring with optical axes; returns cos(axis, direction to ring centre)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    centers = np.stack([c.center for c in cameras]); axes = np.stack([c.R[2] for c in cameras])
    # Capture-volume centre: least-squares closest point to all optical axes.
    A = sum(np.eye(3) - np.outer(a, a) for a in axes); b = sum((np.eye(3) - np.outer(a, a)) @ c for a, c in zip(axes, centers))
    focus = np.linalg.solve(A, b)
    cos = [float(a @ (focus - c) / np.linalg.norm(focus - c)) for a, c in zip(axes, centers)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for k, (i, j) in enumerate(((0, 2), (0, 1))):
        ax[k].scatter(centers[:, i], centers[:, j], c="tab:blue"); ax[k].scatter(*focus[[i, j]], c="tab:red", marker="x")
        for n, (c, a) in enumerate(zip(centers, axes)):
            ax[k].arrow(c[i], c[j], a[i]*0.8, a[j]*0.8, head_width=0.08, color="tab:gray"); ax[k].annotate(f"{n:02d}", c[[i, j]])
        ax[k].set_aspect("equal"); ax[k].set_title(["top (x,z)", "side (x,y)"][k])
    fig.suptitle(f"ring frusta; axis-to-centre cosine min {min(cos):.3f}"); fig.savefig(path, dpi=100); plt.close(fig)
    try:
        import open3d as o3d
        geometry = []
        for c in cameras:
            w2c = np.eye(4); w2c[:3] = np.column_stack((c.R, c.t))
            geometry.append(o3d.geometry.LineSet.create_camera_visualization(c.width, c.height, c.K, w2c, 0.3))
        merged = geometry[0]
        for g in geometry[1:]: merged += g
        o3d.io.write_line_set(str(path.with_suffix(".ply")), merged)
    except Exception as error:  # the matplotlib figure is the required artefact
        print(f"open3d frusta export skipped: {error}")
    return cos


# --------------------------------------------------------------------------- MASt3R helpers
_MATCHER = None


def matcher(cfg: Config):
    global _MATCHER
    if _MATCHER is None:
        from ring_init.geom.match import MASt3RMatcher
        _MATCHER = MASt3RMatcher(cfg.mast3r_model, cfg.device)
    return _MATCHER


def release_matcher() -> None:
    global _MATCHER
    _MATCHER = None; torch.cuda.empty_cache()


def multiview_consistent(xyz: np.ndarray, anchors: tuple[int, int], cameras: list[Camera], images: list[np.ndarray], labels: list[np.ndarray] | None, label_id: int | None, color_l1: float, min_extra: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep points whose colour (and, with labels, instance) agree in >= min_extra non-anchor views.
    Returns keep flag and per-point median colour over agreeing + anchor views."""
    samples, agree = [], []
    for ci, cam in enumerate(cameras):
        uv, z = cam.project(xyz); u = np.rint(uv[:, 0]).astype(int); v = np.rint(uv[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        col = np.zeros((len(xyz), 3), np.float32); col[ok] = images[ci][v[ok], u[ok]] / 255.0
        if labels is not None and label_id is not None:
            lab = np.full(len(xyz), -1); lab[ok] = labels[ci][v[ok], u[ok]]; ok &= lab == label_id
        samples.append(col); agree.append(ok)
    samples = np.stack(samples, 1); agree = np.stack(agree, 1)
    ref = (samples[:, anchors[0]] + samples[:, anchors[1]]) / 2
    close = agree & (np.abs(samples - ref[:, None]).mean(2) < color_l1)
    extra = close.copy(); extra[:, list(anchors)] = False
    keep = extra.sum(1) >= min_extra
    close[:, list(anchors)] = True
    masked = np.where(close[..., None], samples, np.nan)
    return keep, np.nanmedian(masked, 1)


# --------------------------------------------------------------------------- Step 3a: full-frame pairs
def step_match_full(s: Scene) -> None:
    if s.done("match_full"): return
    from ring_init.geom.match import resize_full, resized_to_full, ring_pairs
    t = time.time(); cfg = s.cfg; d = s.dir("match_full"); cams = s.cameras; images = s.images; total = 0
    for a, b in ring_pairs(cfg.camera_count):
        out = d / f"pair_{a:02d}_{b:02d}.npz"
        if out.is_file() and "match_full" not in s.force: total += len(np.load(out)["xyz"]); continue
        ra, sa = resize_full(images[a], cfg.mast3r_size); rb, sb = resize_full(images[b], cfg.mast3r_size)
        uv_a, uv_b, conf = matcher(cfg).match(ra, rb, cfg.matches_per_pair)
        uv_a, uv_b = resized_to_full(uv_a, sa), resized_to_full(uv_b, sb)
        keep = conf >= cfg.pair_confidence
        uv_a, uv_b, conf = uv_a[keep], uv_b[keep], conf[keep]
        xyz = triangulate_dlt(uv_a, uv_b, cams[a], cams[b])
        ok = filter_pair(xyz, uv_a, uv_b, cams[a], cams[b], cfg.reprojection_px, cfg.min_triangulation_angle_deg)
        np.savez_compressed(out, xyz=xyz[ok].astype(np.float32), uv_a=uv_a[ok].astype(np.float32), uv_b=uv_b[ok].astype(np.float32), conf=conf[ok], cam_a=a, cam_b=b)
        total += int(ok.sum()); print(f"  full {a:02d}-{b:02d}: {int(keep.sum()):,} confident -> {int(ok.sum()):,} points")
    s.finish("match_full", {"points": total}, time.time() - t)


def load_full_points(s: Scene) -> tuple[np.ndarray, np.ndarray]:
    xyz, pairs = [], []
    for p in sorted((s.out / "match_full").glob("pair_*.npz")):
        x = np.load(p); xyz.append(x["xyz"]); pairs.append(np.tile([int(x["cam_a"]), int(x["cam_b"])], (len(x["xyz"]), 1)))
    return np.concatenate(xyz).astype(np.float64), np.concatenate(pairs)


# --------------------------------------------------------------------------- Step 2a: floor + persons + scale
def step_floor(s: Scene) -> None:
    if s.done("floor", "floor_plane.json"): return
    t = time.time(); cfg = s.cfg; xyz, _ = load_full_points(s)
    floor = fit_floor(xyz, s.cameras, cfg.floor_ransac_threshold, cfg.floor_ransac_iterations, cfg.court_floor_percentile, cfg.seed)
    floor.save(s.dir("floor") / "floor_plane.json")
    (s.out / "floor" / "timing.json").write_text(json.dumps({"seconds": time.time() - t}))
    print(f"[floor] plane n={np.round(floor.normal, 4)} extent {floor.extent_lo}..{floor.extent_hi}")


def court_filter(boxes: np.ndarray, cam: Camera, floor: Floor, margin: float) -> np.ndarray:
    """Keep boxes whose bottom-centre ray hits the floor inside the (margin-expanded) court extent."""
    if len(boxes) == 0: return np.zeros(0, bool)
    feet = floor.to_plane(floor.ray_hit(cam, np.stack(((boxes[:, 0]+boxes[:, 2])/2, boxes[:, 3]), 1)))
    return np.all((feet >= floor.extent_lo - margin) & (feet <= floor.extent_hi + margin), 1)


def step_persons(s: Scene) -> None:
    if s.done("persons"): return
    from ring_init.masks.segment import SAM2Refiner, torchvision_person_detector
    t = time.time(); cfg = s.cfg; d = s.dir("persons"); cams = s.cameras; floor = Floor.load(s.out / "floor" / "floor_plane.json")
    detector = torchvision_person_detector(cfg.detector, cfg.detector_min_size, cfg.detector_max_size, cfg.detector_score_threshold, cfg.detector_max_detections, cfg.device)
    refiner = SAM2Refiner(cfg.sam2_checkpoint, cfg.sam2_config, cfg.device)
    # Scale is unknown before this step: the court margin is a fraction of the floor extent.
    margin = 0.05 * float(np.max(floor.extent_hi - floor.extent_lo))
    per_cam, counts = [], []
    for ci, (cam, rgb) in enumerate(zip(cams, s.images)):
        boxes = detector(rgb); court = court_filter(boxes, cam, floor, margin); boxes = boxes[court]
        refiner.set_image(rgb)
        masks, scores = refiner.masks(boxes) if len(boxes) else (np.zeros((0, cam.height, cam.width), bool), np.zeros(0))
        keep = masks.reshape(len(masks), -1).sum(1) >= cfg.min_mask_area_px
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
        np.savez_compressed(d / f"cam_{ci:02d}.npz", boxes=boxes, scores=scores, masks=np.packbits(masks, axis=-1), width=cam.width)
        per_cam.append(list(masks)); counts.append({"camera": ci, "court_detections": int(len(boxes)), "all_detections": int(len(court))})
        print(f"  persons cam {ci:02d}: {len(court)} person boxes -> {len(boxes)} on court")
    del detector, refiner; torch.cuda.empty_cache()
    upm = cfg.units_per_meter or estimate_units_per_meter(floor, cams, per_cam, cfg.person_height_m)
    floor.units_per_meter = upm; floor.save(s.out / "floor" / "floor.json")
    (s.out / "floor" / "done.json").write_text(json.dumps({"units_per_meter": upm}) + "\n")
    s.finish("persons", {"units_per_meter": upm, "cameras": counts}, time.time() - t)


def load_persons(s: Scene, ci: int) -> dict:
    x = np.load(s.out / "persons" / f"cam_{ci:02d}.npz")
    masks = np.unpackbits(x["masks"], axis=-1, count=int(x["width"])).astype(bool) if len(x["boxes"]) else np.zeros((0, 1, 1), bool)
    return {"boxes": x["boxes"], "masks": masks, "union": masks.any(0) if len(masks) else None}


def scaled(cfg: Config, s: Scene) -> Config:
    """Config with units_per_meter filled from the floor step."""
    if cfg.units_per_meter is None: cfg = replace(cfg, units_per_meter=s.floor().units_per_meter)
    return cfg


# --------------------------------------------------------------------------- Step 2b: instances
def step_instances(s: Scene) -> None:
    if s.done("instances"): return
    from ring_init.masks.instances import find_instances, label_cameras
    from ring_init.masks.segment import SAM2Refiner, save_label_masks
    t = time.time(); cfg = scaled(s.cfg, s); d = s.dir("instances"); cams = s.cameras; floor = s.floor()
    person = [load_persons(s, ci) for ci in range(len(cams))]
    unions = [p["union"] if p["union"] is not None else np.zeros((c.height, c.width), bool) for p, c in zip(person, cams)]
    vox = cfg.m(cfg.occupancy_voxel_m)
    instances, stats = find_instances(cams, unions, floor, vox, cfg.m(cfg.occupancy_height_m), cfg.m(cfg.court_margin_m), cfg.occupancy_min_views, cfg.occupancy_allowed_misses, cfg.m(cfg.instance_min_height_m), cfg.instance_min_voxels, cfg.m(cfg.instance_split_footprint_m), cfg.device)
    if not instances: raise RuntimeError(f"No person instances found by occupancy carving: {stats}")
    # Voxel footprint in pixels (for splatting the projected occupancy silhouette).
    dist = float(np.median([np.linalg.norm(c.center - floor.origin) for c in cams])); f = float(np.median([c.K[0, 0] for c in cams]))
    radius = int(np.ceil(vox * f / dist)) + cfg.instance_box_padding_px
    refiner = SAM2Refiner(cfg.sam2_checkpoint, cfg.sam2_config, cfg.device)
    labels, depths, report = label_cameras(instances, cams, person, refiner, s.images, radius, cfg.instance_min_iou, cfg.min_mask_area_px, cfg.instance_max_hist_distance)
    del refiner; torch.cuda.empty_cache()
    save_label_masks(labels, d / "masks")
    np.savez_compressed(d / "occupancy.npz", **{f"instance_{i.instance_id:03d}": i.voxels for i in instances})
    info = {"instances": [{"instance_id": i.instance_id, "floor_uv": i.floor_uv.tolist(), "height_m": i.height / cfg.units_per_meter, "voxels": len(i.voxels), "center": i.center.tolist(), "views": [r["camera"] for r in report if i.instance_id in r["labelled"]]} for i in instances],
            "depths": [{str(k): v for k, v in dd.items()} for dd in depths], "cameras": report, "occupancy": stats, "voxel": vox}
    (d / "instances.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    contact_sheet(s.images, labels, d / "contact_sheet.jpg")
    s.finish("instances", {"instances": len(instances), "views_per_instance": [len(x["views"]) for x in info["instances"]]}, time.time() - t)


def contact_sheet(images: list[np.ndarray], labels: list[np.ndarray], path: Path, width: int = 640) -> None:
    rng = np.random.default_rng(3); palette = rng.integers(60, 255, (256, 3)).astype(np.uint8); palette[0] = 0
    tiles = []
    for ci, (im, lab) in enumerate(zip(images, labels)):
        over = im.copy(); m = lab > 0; over[m] = (0.35*im[m] + 0.65*palette[lab[m] % 256]).astype(np.uint8)
        for k in np.unique(lab[lab > 0]):
            ys, xs = np.nonzero(lab == k); cv2.putText(over, str(int(k)), (int(xs.mean()) - 8, int(ys.min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tile = cv2.resize(over, (width, int(round(width * images[0].shape[0] / images[0].shape[1]))))
        cv2.putText(tile, f"cam {ci:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); tiles.append(tile)
    rows = [np.concatenate(tiles[i:i+4], 1) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def static_points(s: Scene, cfg: Config, labels: list[np.ndarray]) -> np.ndarray:
    """Full-frame points that are background-labelled and colour-consistent in >= 2 extra views
    (spurious DLT points would otherwise act as fake occluders)."""
    xyz, pairs = load_full_points(s); keep = np.zeros(len(xyz), bool)
    for a, b in {tuple(p) for p in pairs.tolist()}:
        sel = np.flatnonzero((pairs[:, 0] == a) & (pairs[:, 1] == b))
        keep[sel] = multiview_consistent(xyz[sel], (a, b), s.cameras, s.images, labels, 0, cfg.multi_view_color_l1, cfg.occluder_min_extra_views)[0]
    # The floor occludes only what is below it (clipped separately); as an occluder it would
    # un-carve a skirt of voxels around every foot.
    keep &= s.floor().height(xyz) > cfg.m(cfg.occluder_min_height_m)
    return xyz[keep]


# --------------------------------------------------------------------------- Step 4: hulls
def step_hulls(s: Scene) -> None:
    if s.done("hulls"): return
    t = time.time(); cfg = scaled(s.cfg, s); d = s.dir("hulls"); cams = s.cameras; info = s.instance_info(); labels = s.labels(); floor = s.floor()
    depths = [{int(k): v for k, v in dd.items()} for dd in info["depths"]]
    occ = np.load(s.out / "instances" / "occupancy.npz"); vox = cfg.m(cfg.visual_hull_voxel_m); coarse = info["voxel"]; summary = {}
    from ring_init.geom.hull import occluder_depth_maps
    occluders = occluder_depth_maps(static_points(s, cfg, labels), cams, cfg.occluder_splat_px)
    for inst in info["instances"]:
        k = inst["instance_id"]; pts = occ[f"instance_{k:03d}"]
        lo, hi = pts.min(0) - 2*coarse, pts.max(0) + 2*coarse
        centres, surface = instance_hull(lo, hi, vox, cams, labels, k, depths, occluders, cfg.m(cfg.occluder_margin_m), cfg.hull_allowed_misses, cfg.device)
        above = floor.height(centres) >= -0.5*vox   # the floor occludes everything below it
        centres, surface = centres[above], surface[above]
        np.savez_compressed(d / f"instance_{k:03d}.npz", occupied=centres.astype(np.float32), surface=surface, voxel=vox)
        summary[k] = {"occupied": int(len(centres)), "surface": int(surface.sum())}
        print(f"  hull {k:03d}: {summary[k]}")
    s.finish("hulls", {"voxel": vox, "instances": summary}, time.time() - t)


def load_hull(s: Scene, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.load(s.out / "hulls" / f"instance_{k:03d}.npz"); return x["occupied"].astype(np.float64), x["surface"], float(x["voxel"])


# --------------------------------------------------------------------------- Step 3b: per-instance crops
def step_match_crops(s: Scene) -> None:
    if s.done("match_crops"): return
    from ring_init.geom.hull import dilate
    from ring_init.geom.match import ring_pairs
    t = time.time(); cfg = scaled(s.cfg, s); d = s.dir("match_crops"); cams = s.cameras; images = s.images; labels = s.labels(); info = s.instance_info()
    size = cfg.mast3r_size; summary: dict[int, dict] = {}
    for inst in info["instances"]:
        k = inst["instance_id"]; occupied, _, vox = load_hull(s, k); stats = {"pairs": 0, "matches": 0, "geometric": 0, "in_hull": 0}
        masks = [dilate(lab == k, 2) for lab in labels]
        for a, b in ring_pairs(cfg.camera_count):
            out = d / f"instance_{k:03d}_pair_{a:02d}_{b:02d}.npz"
            if not (masks[a].sum() >= cfg.min_mask_area_px and masks[b].sum() >= cfg.min_mask_area_px): continue
            boxes = []
            for c in (a, b):
                ys, xs = np.nonzero(labels[c] == k)
                boxes.append(square_box(np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()]), cfg.crop_padding_fraction, cfg.crop_min_size_px, cams[c].width, cams[c].height))
            crops = [cv2.resize(images[c][bx[1]:bx[3], bx[0]:bx[2]], (size, size), interpolation=cv2.INTER_CUBIC) for c, bx in zip((a, b), boxes)]
            seed_mask = cv2.resize(masks[a][boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]].astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
            sy, sx = np.nonzero(seed_mask[::cfg.crop_seed_stride, ::cfg.crop_seed_stride])
            uv_a, uv_b, conf = matcher(cfg).match(crops[0], crops[1], size*size // 4, seeds=(sx*cfg.crop_seed_stride, sy*cfg.crop_seed_stride))
            uv_a, uv_b = from_crop(uv_a, boxes[0], size), from_crop(uv_b, boxes[1], size)
            ia, ib = np.rint(uv_a).astype(int), np.rint(uv_b).astype(int)
            ok = (conf >= cfg.pair_confidence) & masks[a][ia[:, 1].clip(0, cams[a].height-1), ia[:, 0].clip(0, cams[a].width-1)] & masks[b][ib[:, 1].clip(0, cams[b].height-1), ib[:, 0].clip(0, cams[b].width-1)]
            uv_a, uv_b, conf = uv_a[ok], uv_b[ok], conf[ok]; stats["matches"] += int(ok.sum())
            xyz = triangulate_dlt(uv_a, uv_b, cams[a], cams[b])
            geo = filter_pair(xyz, uv_a, uv_b, cams[a], cams[b], cfg.reprojection_px, cfg.min_triangulation_angle_deg); stats["geometric"] += int(geo.sum())
            carve = geo & inside_hull(xyz, occupied, vox, cfg.hull_carve_dilation_voxels); stats["in_hull"] += int(carve.sum())
            np.savez_compressed(out, xyz=xyz[carve].astype(np.float32), uv_a=uv_a[carve].astype(np.float32), uv_b=uv_b[carve].astype(np.float32), conf=conf[carve], cam_a=a, cam_b=b, n_geometric=int(geo.sum()))
            stats["pairs"] += 1
        summary[k] = stats; print(f"  crops {k:03d}: {stats}")
    release_matcher()
    s.finish("match_crops", {"instances": summary}, time.time() - t)


# --------------------------------------------------------------------------- Step 5: fuse
def visible_median_colors(points: np.ndarray, cameras: list[Camera], images: list[np.ndarray], labels: list[np.ndarray], label_id: int, depth_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Median colour over views where the point is in frustum, carries the label, and passes a
    z-buffer test against the point set itself. Also returns mean unit direction to those cameras."""
    cols, dirs = [], np.zeros_like(points); cnt = np.zeros(len(points))
    for ci, cam in enumerate(cameras):
        uv, z = cam.project(points); u = np.rint(uv[:, 0]).astype(int); v = np.rint(uv[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        flat = np.where(ok, v*cam.width + u, 0); zbuf = np.full(cam.width*cam.height, np.inf); np.minimum.at(zbuf, flat[ok], z[ok])
        ok &= z <= zbuf[flat] + depth_tol
        lab = np.full(len(points), -1); lab[ok] = labels[ci][v[ok], u[ok]]; ok &= lab == label_id
        c = np.full((len(points), 3), np.nan); c[ok] = images[ci][v[ok], u[ok]] / 255.0; cols.append(c)
        dv = cam.center - points; dv /= np.linalg.norm(dv, axis=1, keepdims=True); dirs[ok] += dv[ok]; cnt += ok
    with np.errstate(all="ignore"): med = np.nanmedian(np.stack(cols, 1), 1)
    return med, dirs


def fuse_cloud(points: np.ndarray, colors: np.ndarray, voxel: float, cfg: Config, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud.normals = o3d.utility.Vector3dVector(dirs)  # carried through voxel averaging, used for orientation
    if len(points) > cfg.outlier_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=cfg.outlier_neighbors, std_ratio=cfg.outlier_std_ratio)
    cloud = cloud.voxel_down_sample(voxel)
    view_dir = np.asarray(cloud.normals).copy()
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.normal_radius_voxels*voxel, max_nn=30))
    n = np.asarray(cloud.normals); flip = (n * view_dir).sum(1) < 0; n[flip] *= -1
    return np.asarray(cloud.points), np.asarray(cloud.colors), n


def instance_triangulated(s: Scene, cfg: Config, k: int, labels: list[np.ndarray]) -> np.ndarray:
    """Hull-carved crop-MASt3R points of instance k that pass the multi-view consistency check."""
    tri = []
    for p in sorted((s.out / "match_crops").glob(f"instance_{k:03d}_pair_*.npz")):
        x = np.load(p)
        if len(x["xyz"]) == 0: continue
        keep, _ = multiview_consistent(x["xyz"].astype(np.float64), (int(x["cam_a"]), int(x["cam_b"])), s.cameras, s.images, labels, k, cfg.multi_view_color_l1, cfg.multi_view_min_extra_views)
        tri.append(x["xyz"][keep].astype(np.float64))
    return np.concatenate(tri) if tri else np.zeros((0, 3))


def triangulated_only_cloud(s: Scene, cfg: Config, k: int, labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ablation seed without hull fill: triangulated points only, fused the same way."""
    _, _, vox = load_hull(s, k); tri = instance_triangulated(s, cfg, k, labels)
    colors, dirs = visible_median_colors(tri, s.cameras, s.images, labels, k, 2*vox); ok = np.isfinite(colors).all(1)
    return fuse_cloud(tri[ok], colors[ok], cfg.m(cfg.player_voxel_m), cfg, dirs[ok])


def step_fuse(s: Scene) -> None:
    if s.done("fuse"): return
    from ring_init.geom.fuse import write_ply
    from scipy.spatial import cKDTree
    t = time.time(); cfg = scaled(s.cfg, s); d = s.dir("fuse"); cams = s.cameras; images = s.images; labels = s.labels(); info = s.instance_info(); summary = {}
    for inst in info["instances"]:
        k = inst["instance_id"]; occupied, surface, vox = load_hull(s, k)
        tri = instance_triangulated(s, cfg, k, labels)
        hull_pts = occupied[surface]
        if len(tri):
            near = cKDTree(tri).query(hull_pts, k=1)[0]; fill = hull_pts[near > cfg.hull_fill_distance_voxels * vox]
        else: fill = hull_pts
        pts = np.concatenate((tri, fill)); src = np.r_[np.zeros(len(tri)), np.ones(len(fill))]
        colors, dirs = visible_median_colors(pts, cams, images, labels, k, 2*vox)
        valid = np.isfinite(colors).all(1)
        pts, colors, dirs, src = pts[valid], colors[valid], dirs[valid], src[valid]
        fused, fcol, normals = fuse_cloud(pts, colors, cfg.m(cfg.player_voxel_m), cfg, dirs)
        write_ply(d / f"instance_{k:03d}.ply", fused, fcol, normals)
        write_ply(d / f"instance_{k:03d}_triangulated.ply", tri, np.tile([[1., .3, .3]], (len(tri), 1)))
        summary[k] = {"triangulated": int(len(tri)), "hull_surface": int(len(hull_pts)), "hull_fill": int(len(fill)), "colored": int(valid.sum()), "fused": int(len(fused))}
        print(f"  fuse {k:03d}: {summary[k]}")
    # Background: full-frame points outside every (margin-dilated) person hull, background label in anchors.
    xyz, pairs = load_full_points(s); person = np.zeros(len(xyz), bool)
    for inst in info["instances"]:
        occupied, _, vox = load_hull(s, inst["instance_id"])
        person |= inside_hull(xyz, occupied, vox, int(np.ceil(cfg.m(cfg.background_hull_margin_m) / vox)))
    keep = ~person; bg_keep = np.zeros(len(xyz), bool); bg_col = np.zeros((len(xyz), 3))
    for a, b in {tuple(p) for p in pairs.tolist()}:
        sel = np.flatnonzero((pairs[:, 0] == a) & (pairs[:, 1] == b) & keep)
        ok, col = multiview_consistent(xyz[sel], (a, b), cams, images, labels, 0, cfg.multi_view_color_l1, cfg.multi_view_min_extra_views)
        bg_keep[sel] = ok; bg_col[sel] = col
    pts, cols = xyz[bg_keep], bg_col[bg_keep]
    dirs = np.zeros_like(pts)
    for cam in cams:
        _, z = cam.project(pts); dv = cam.center - pts; dirs += np.where((z > 0)[:, None], dv / np.linalg.norm(dv, axis=1, keepdims=True), 0)
    fused, fcol, normals = fuse_cloud(pts, np.nan_to_num(cols), cfg.m(cfg.background_voxel_m), cfg, dirs)
    write_ply(d / "background.ply", fused, fcol, normals)
    summary[0] = {"full_frame": int(len(xyz)), "outside_person_hulls": int(keep.sum()), "multiview": int(bg_keep.sum()), "fused": int(len(fused))}
    print(f"  fuse background: {summary[0]}")
    s.finish("fuse", {"instances": summary}, time.time() - t)


GEOMETRY_STEPS = {"calib": step_calib, "match_full": step_match_full, "floor": step_floor, "persons": step_persons,
                  "instances": step_instances, "hulls": step_hulls, "match_crops": step_match_crops, "fuse": step_fuse}


def run_geometry(s: Scene, until: str | None = None) -> None:
    s.cfg.save(s.out / "resolved_config.json")
    for name, fn in GEOMETRY_STEPS.items():
        fn(s)
        if name == until: break


# --------------------------------------------------------------------------- Step 7/8: training
def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d
    c = o3d.io.read_point_cloud(str(path)); return np.asarray(c.points), np.asarray(c.colors), np.asarray(c.normals)


def sparse_depth_map(points: np.ndarray, cam: Camera, valid: np.ndarray | None = None) -> np.ndarray:
    """Nearest fused-point depth per pixel (0 = no sample), optionally restricted to `valid` pixels."""
    uv, z = cam.project(points); u = np.rint(uv[:, 0]).astype(int); v = np.rint(uv[:, 1]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    buf = np.full(cam.height*cam.width, np.inf, np.float32); np.minimum.at(buf, v[ok]*cam.width + u[ok], z[ok])
    buf[~np.isfinite(buf)] = 0; buf = buf.reshape(cam.height, cam.width)
    if valid is not None: buf[~valid] = 0
    return buf


def _base_view(cam: Camera, rgb: np.ndarray, device: str, name: str):
    from ring_init.gs.train import View
    w2c = np.eye(4, dtype=np.float32); w2c[:3] = np.column_stack((cam.R, cam.t))
    z = torch.zeros((cam.height, cam.width), dtype=torch.bool, device=device)
    return View(image=torch.as_tensor(rgb, device=device).permute(2, 0, 1).float() / 255, loss_mask=z, alpha_target=z.float(), alpha_ignore=z,
                viewmat=torch.as_tensor(w2c, device=device), K=torch.as_tensor(cam.K, dtype=torch.float32, device=device), width=cam.width, height=cam.height, name=name)


def instance_views(s: Scene, k: int, points: np.ndarray, labels: list[np.ndarray], depths: list[dict[int, float]], exclude_cameras: tuple[int, ...] = ()) -> list:
    """Cropped training views of instance k. All crops share one (tile-aligned) size so the
    trainer can rasterize several cameras in a single batched call."""
    from dataclasses import replace as dc_replace
    from ring_init.gs.train import TILE, crop_view
    cfg = s.cfg; dev = cfg.device; full, boxes = [], []
    for ci, (cam, lab) in enumerate(zip(s.cameras, labels)):
        if ci in exclude_cameras or k not in depths[ci]: continue
        own = lab == k
        nearer = [j for j, d in depths[ci].items() if j != k and d < depths[ci][k]]
        ignore = np.isin(lab, nearer) if nearer else np.zeros_like(own)
        # Where the hull projects onto background, this view missed it (static occlusion or a
        # mask error): alpha there is unknown, not zero.
        ignore |= hull_silhouette(s, k, cam) & (lab == 0)
        v = dc_replace(_base_view(cam, s.images[ci], dev, f"cam_{ci:02d}"), loss_mask=torch.as_tensor(own, device=dev), alpha_target=torch.as_tensor(own, device=dev).float(),
                       alpha_ignore=torch.as_tensor(ignore, device=dev), sparse_depth=torch.as_tensor(sparse_depth_map(points, cam, own), device=dev))
        uv, z = cam.project(points); uv = uv[z > 0]
        ys, xs = np.nonzero(own)
        lo = np.minimum(uv.min(0), [xs.min(), ys.min()]) - cfg.crop_render_padding_px; hi = np.maximum(uv.max(0), [xs.max(), ys.max()]) + cfg.crop_render_padding_px
        full.append((v, cam)); boxes.append((lo, hi))
    if not full: return []
    w = int(np.ceil(max(h[0] - l[0] for l, h in boxes) / TILE) * TILE); h_ = int(np.ceil(max(h[1] - l[1] for l, h in boxes) / TILE) * TILE)
    views = []
    for (v, cam), (lo, hi) in zip(full, boxes):
        cw, ch = min(w, cam.width // TILE * TILE), min(h_, cam.height // TILE * TILE)
        c = (lo + hi) / 2
        x0 = int(np.clip(np.floor((c[0] - cw/2) / TILE) * TILE, 0, (cam.width - cw) // TILE * TILE))
        y0 = int(np.clip(np.floor((c[1] - ch/2) / TILE) * TILE, 0, (cam.height - ch) // TILE * TILE))
        views.append(crop_view(v, (x0, y0, x0 + cw, y0 + ch)))
    return views


def hull_silhouette(s: Scene, k: int, cam: Camera) -> np.ndarray:
    from ring_init.masks.instances import splat_silhouette
    occupied, _, _ = load_hull(s, k)
    return splat_silhouette(occupied, cam, 0)[0]


def person_exclusion(s: Scene, labels: list[np.ndarray]) -> list[np.ndarray]:
    """Pixels the background must not explain. With the frozen person models composited in
    front (bg_composite_persons), labelled person pixels stay supervised: a background Gaussian
    drawn in front of a person is penalized, one behind an opaque person gets no gradient.
    Detector person pixels without an instance (unmodelled people) are always excluded."""
    from ring_init.geom.hull import dilate
    result = []
    for ci, lab in enumerate(labels):
        union = load_persons(s, ci)["union"]
        ex = np.zeros_like(lab, bool) if union is None else union.copy()
        if s.cfg.bg_composite_persons: ex &= lab == 0
        else: ex |= lab > 0
        result.append(dilate(ex, s.cfg.bg_person_mask_dilation_px))
    return result


def step_train(s: Scene, exclude_cameras: tuple[int, ...] = (), tag: str = "train") -> None:
    """Per-instance t0 optimization, then the background. `exclude_cameras` supports
    leave-one-camera-out runs (tag keeps their outputs separate)."""
    from dataclasses import replace as dc_replace
    from ring_init.gs.export import save_ply
    from ring_init.gs.init_gaussians import initialize_gaussians
    from ring_init.gs.train import train_gaussians
    cfg = scaled(s.cfg, s); d = s.dir(tag); info = s.instance_info(); labels = s.labels()
    depths = [{int(k): v for k, v in dd.items()} for dd in info["depths"]]
    stats = json.loads((d / "stats.json").read_text()) if (d / "stats.json").is_file() else {}
    for inst in info["instances"]:
        k = inst["instance_id"]; target = d / f"instance_{k:03d}" / "point_cloud.ply"
        if target.is_file() and tag not in s.force: continue
        pts, cols, normals = _read_ply(s.out / "fuse" / f"instance_{k:03d}.ply") if cfg.use_hull else triangulated_only_cloud(s, cfg, k, labels)
        model = initialize_gaussians(pts, cols, normals, k, cfg)
        views = instance_views(s, k, pts, labels, depths, exclude_cameras)
        if len(views) < 3: print(f"  skip instance {k}: {len(views)} views"); continue
        torch.manual_seed(cfg.seed + k)
        from ring_init.geom.hull import HullMembership
        occupied, _, vox = load_hull(s, k)
        hull = HullMembership(occupied, vox, cfg.hull_prune_dilation_voxels, cfg.device)
        result = train_gaussians(model, views, cfg, "instance", target.parent / "train_log.jsonl", extra_prune=hull.outside if cfg.use_hull else None)
        save_ply(model, target); render_sheet(model, views, target.parent / "train_views.jpg")
        stats[str(k)] = {**{a: b for a, b in result.items() if a != "densify_history"}, "initial": int(len(pts)), "views": len(views)}
        (d / "stats.json").write_text(json.dumps(stats, indent=2, default=float) + "\n")
        print(f"  train {k:03d}: {len(pts)} -> {result['final_gaussians']} gaussians, {result['wall_time_s']:.1f}s")
        del model, views; torch.cuda.empty_cache()
    target = d / "instance_000" / "point_cloud.ply"
    if cfg.background and (not target.is_file() or tag in s.force):
        pts, cols, normals = _read_ply(s.out / "fuse" / "background.ply")
        model = initialize_gaussians(pts, cols, normals, 0, cfg)
        exclusion = person_exclusion(s, labels); views = []
        for ci, cam in enumerate(s.cameras):
            if ci in exclude_cameras: continue
            keep = ~exclusion[ci]; dev = cfg.device
            views.append(dc_replace(_base_view(cam, s.images[ci], dev, f"cam_{ci:02d}"), loss_mask=torch.as_tensor(keep, device=dev), sparse_depth=torch.as_tensor(sparse_depth_map(pts, cam, keep), device=dev)))
        frozen = None
        if cfg.bg_composite_persons:
            from ring_init.gs.export import concat_models, load_ply
            persons = [load_ply(p, cfg.device) for p in sorted(d.glob("instance_*/point_cloud.ply")) if p.parent.name != "instance_000"]
            frozen = concat_models(persons) if persons else None
        from ring_init.geom.hull import HullMembership
        hulls = [HullMembership(*load_hull(s, inst["instance_id"])[::2], int(np.ceil(cfg.m(cfg.background_hull_margin_m) / load_hull(s, inst["instance_id"])[2])), cfg.device) for inst in info["instances"]]
        def inside_any_person(means: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(len(means), dtype=torch.bool, device=means.device)
            for h in hulls: out |= h.inside(means)
            return out
        result = train_gaussians(model, views, cfg, "background", target.parent / "train_log.jsonl", frozen=frozen, extra_prune=inside_any_person)
        save_ply(model, target)
        stats["0"] = {**{a: b for a, b in result.items() if a != "densify_history"}, "initial": int(len(pts)), "views": len(views)}
        (d / "stats.json").write_text(json.dumps(stats, indent=2, default=float) + "\n")
        print(f"  train background: {len(pts)} -> {result['final_gaussians']} gaussians, {result['wall_time_s']:.1f}s")
        del model, views; torch.cuda.empty_cache()
    total = sum(v["wall_time_s"] for v in stats.values())
    timings = json.loads(s.timings_path.read_text()) if s.timings_path.is_file() else {}
    timings[tag] = total; s.timings_path.write_text(json.dumps(timings, indent=2) + "\n")


@torch.no_grad()
def render_sheet(model, views: list, path: Path, max_views: int = 12) -> None:
    """Per view: target crop | mask | render | alpha | |residual| (rows)."""
    from ring_init.gs.train import render
    rows = []
    for v in views[:max_views]:
        rgb, _, alpha, _ = render(model, v.viewmat, v.K, v.width, v.height)
        tgt = v.image.permute(1, 2, 0); m = v.alpha_target[..., None].expand(-1, -1, 3)
        res = (rgb - tgt).abs() * v.loss_mask[..., None]
        tiles = [tgt, m, rgb, alpha[..., None].expand(-1, -1, 3), res * 3]
        row = torch.cat([t.clamp(0, 1) for t in tiles], 1).cpu().numpy()
        h = 160; row = cv2.resize((row*255).astype(np.uint8), (int(row.shape[1]*h/row.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        rows.append(row)
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 2), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- Step 9: export
def step_export(s: Scene, tag: str = "train") -> None:
    import hashlib, shutil
    cfg = scaled(s.cfg, s); src = s.out / tag; dst = s.dir("canonical"); info = s.instance_info()
    entries = []
    for inst_dir in sorted(src.glob("instance_*")):
        k = int(inst_dir.name.split("_")[-1]); out = dst / inst_dir.name; out.mkdir(parents=True, exist_ok=True)
        for f in ("point_cloud.ply", "instance_ids.npy"): shutil.copy2(inst_dir / f, out / f)
        ids = np.load(out / "instance_ids.npy")
        if not np.all(ids == k): raise RuntimeError(f"{inst_dir}: instance_ids.npy does not match directory id {k}")
        entries.append({"instance_id": k, "kind": "background" if k == 0 else "person", "model": str(out / "point_cloud.ply"), "gaussians": int(len(ids)),
                        "masks": None if k == 0 else str(s.out / "instances" / "masks" / "cam_XX" / f"instance_{k:03d}.png")})
    calib = s.source / cfg.calibration
    manifest = {"scene": s.name, "stage": "A", "frame": 0, "units_per_meter": cfg.units_per_meter, "calibration": str(calib),
                "calibration_sha256": hashlib.sha256(calib.read_bytes()).hexdigest(), "instances": entries,
                "config": json.loads((s.out / "resolved_config.json").read_text()), "immutable": True,
                "note": "Canonical rest models. Stage B writes control-point state only and never overwrites these files."}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[export] {len(entries)} models -> {dst}")
