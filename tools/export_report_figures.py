#!/usr/bin/env python3
"""Export the report's method figures from a completed CourtFlow-GS run.

The defaults point at the diagnostic run used for the accompanying report.  The script is
deliberately data-only: it reads cached run artifacts and never changes a reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PALETTE = np.array([
    (0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0),
], np.uint8)


def read_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def save(path: Path, im: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def labels(root: Path, cam: int) -> np.ndarray:
    paths = sorted((root / "instances" / "masks" / f"cam_{cam:02d}").glob("instance_*.png"))
    if not paths:
        raise FileNotFoundError(f"No labels for camera {cam}")
    out = np.zeros(cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE).shape, np.int32)
    for p in paths:
        ident = int(p.stem.split("_")[-1])
        out[cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0] = ident
    return out


def crop_for_foreground(im: np.ndarray, lab: np.ndarray, pad: int = 50) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(lab > 0)
    x0, x1 = max(0, xs.min() - pad), min(im.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(im.shape[0], ys.max() + pad)
    return im[y0:y1, x0:x1], (x0, y0, x1, y1)


def overlay_masks(im: np.ndarray, lab: np.ndarray) -> np.ndarray:
    out = im.copy()
    for ident in np.unique(lab):
        if ident == 0:
            continue
        m = lab == ident
        col = PALETTE[ident % len(PALETTE)]
        out[m] = (0.42 * out[m] + 0.58 * col).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs):
            cv2.putText(out, str(ident), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def square_box(m: np.ndarray, frac: float, min_size: int, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(m)
    lo, hi = np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()])
    side = int(np.ceil(max(np.max(hi - lo) * (1 + 2 * frac), min_size)))
    side = min(side, width, height)
    ctr = (lo + hi) / 2
    x0 = int(np.clip(round(ctr[0] - side / 2), 0, width - side))
    y0 = int(np.clip(round(ctr[1] - side / 2), 0, height - side))
    return x0, y0, x0 + side, y0 + side


def crop_matches(root: Path, images: list[np.ndarray], labs: list[np.ndarray], out: Path) -> None:
    files = list((root / "match_crops").glob("instance_*_pair_*.npz"))
    best = max(files, key=lambda p: len(np.load(p)["xyz"]))
    x = np.load(best)
    a, b = int(x["cam_a"]), int(x["cam_b"])
    ident = int(best.stem.split("_")[1])
    ba = square_box(labs[a] == ident, .35, 96, images[a].shape[1], images[a].shape[0])
    bb = square_box(labs[b] == ident, .35, 96, images[b].shape[1], images[b].shape[0])
    ca, cb = images[a][ba[1]:ba[3], ba[0]:ba[2]], images[b][bb[1]:bb[3], bb[0]:bb[2]]
    ca, cb = cv2.resize(ca, (512, 512)), cv2.resize(cb, (512, 512))
    canvas = np.concatenate((ca, cb), axis=1)
    rng = np.random.default_rng(5)
    n = min(150, len(x["uv_a"]))
    idx = rng.choice(len(x["uv_a"]), n, replace=False)
    for i in idx:
        ua, ub = x["uv_a"][i], x["uv_b"][i]
        pa = ((ua - np.array(ba[:2])) * 512 / (ba[2] - ba[0])).astype(int)
        pb = ((ub - np.array(bb[:2])) * 512 / (bb[2] - bb[0])).astype(int) + np.array([512, 0])
        col = tuple(int(v) for v in rng.integers(80, 255, 3))
        cv2.line(canvas, tuple(pa), tuple(pb), col, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pa), 2, col, -1)
        cv2.circle(canvas, tuple(pb), 2, col, -1)
    cv2.putText(canvas, f"instance {ident}: cameras {a:02d}--{b:02d}; {n} crop MASt3R matches",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    save(out, canvas)


def strict_hull(root: Path, ident: int, voxel: float) -> np.ndarray:
    """Recompute a conventional hull: every in-frustum view must show the person label."""
    from ring_init.geom.hull import project_torch
    from ring_init.io.calib import load_json
    info = json.loads((root / "instances" / "instances.json").read_text())
    coarse = np.load(root / "instances" / "occupancy.npz")[f"instance_{ident:03d}"]
    coarse_voxel = float(info["voxel"])
    lo, hi = coarse.min(0) - 2 * coarse_voxel, coarse.max(0) + 2 * coarse_voxel
    axes = [torch.arange(lo[d], hi[d] + voxel * .5, voxel, device="cuda", dtype=torch.float64) for d in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    seen = torch.zeros(len(grid), dtype=torch.int32, device="cuda")
    misses = torch.zeros_like(seen)
    for ci, cam in enumerate(load_json(root / "calib" / "calibration_undistorted.json")):
        lab = torch.as_tensor(labels(root, ci), device="cuda")
        u, v, _, inside = project_torch(grid, cam)
        seen += inside.int(); misses += (inside & (lab[v, u] != ident)).int()
    return grid[(seen >= 2) & (misses == 0)].cpu().numpy()


def hull_figure(root: Path, out: Path, ident: int = 1) -> None:
    """Compare a recomputed strict hull to the stored occlusion-aware hull."""
    h = np.load(root / "hulls" / f"instance_{ident:03d}.npz")
    aware = h["occupied"][h["surface"]]
    strict = strict_hull(root, ident, float(h["voxel"]))
    rng = np.random.default_rng(1)
    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    strict_title = "Strict silhouette intersection (empty)" if len(strict) == 0 else "Strict silhouette intersection"
    for n, (title, points) in enumerate(((strict_title, strict),
                                         ("Occlusion-aware visual hull", aware)), 1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        p = points[rng.choice(len(points), min(12000, len(points)), replace=False)] if len(points) else points
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=.35, c="#3b82f6" if n == 1 else "#f97316")
        ax.set_title(title); ax.set_axis_off(); ax.view_init(20, -65); ax.set_box_aspect((1, 1, 1.5))
    fig.suptitle(f"Player {ident}: depth-explained occlusion preserves hidden surface")
    fig.savefig(out, dpi=220); plt.close(fig)


def controls_figure(root: Path, out: Path) -> None:
    from plyfile import PlyData
    cp = np.load(root / "stage_b_diag_crop_ball" / "control_positions.npz")
    pos, group = cp["pos"][0], cp["group"]
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    # Subsample canonical splats so controls remain readable while retaining their spatial context.
    clouds = []
    for ply in (root / "canonical").glob("instance_*/point_cloud.ply"):
        v = PlyData.read(str(ply))["vertex"].data
        clouds.append(np.column_stack((v["x"], v["y"], v["z"])))
    splats = np.concatenate(clouds)
    take = np.linspace(0, len(splats) - 1, min(50000, len(splats))).astype(int)
    ax.scatter(splats[take, 0], splats[take, 1], splats[take, 2], s=.18, color="#9ca3af", alpha=.28,
               label="canonical splats")
    ids = np.unique(group)
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(ids):
        p = pos[group == g]
        take = np.linspace(0, len(p) - 1, min(300, len(p))).astype(int)
        ax.scatter(p[take, 0], p[take, 1], p[take, 2], s=5, color=cmap(i % 20), label=f"group {g}")
    ax.set_title("Stage-B controls, coloured by deformation group")
    ax.set_axis_off(); ax.view_init(24, -62); ax.legend(loc="upper left", fontsize=6, ncol=2)
    fig.savefig(out, dpi=220); plt.close(fig)


def prompt_figure(root: Path, out: Path, frame: int = 150, cam: int = 11) -> None:
    image = read_rgb(root / "frames" / f"cam_{cam:02d}" / f"{frame:06d}.png")
    lab = labels(root, cam) if frame == 0 else None
    # Diagnostic masks are saved as the current frame labels in video_masks when available.
    candidates = list((root / "video_masks" / f"cam_{cam:02d}").glob(f"*{frame:06d}*"))
    if candidates:
        raw = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        lab = raw.astype(np.int32)
    if lab is None:
        # Use visible-frame prompt boxes over the saved diagnostic image when masks were not cached.
        lab = np.zeros(image.shape[:2], np.int32)
    over = overlay_masks(image, lab)
    for ident in np.unique(lab):
        if ident == 0: continue
        ys, xs = np.where(lab == ident)
        cv2.rectangle(over, (xs.min(), ys.min()), (xs.max(), ys.max()), tuple(map(int, PALETTE[ident % len(PALETTE)])), 2)
    cv2.putText(over, "SAM2 refresh: rendered prompt boxes + instance masks", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 3, cv2.LINE_AA)
    save(out, over)


def video_frame(path: Path, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Cannot decode frame {frame} of {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("/media/storage/peripheral_frame0_1080p/ring_init_out_f0/basketball"))
    ap.add_argument("--out", type=Path, default=Path("report/figures/method"))
    args = ap.parse_args(); root, out = args.run_root, args.out
    images = [read_rgb(root / "calib" / f"cam_{i:02d}.png") for i in range(12)]
    labs = [labels(root, i) for i in range(12)]
    image, lab = images[0], labs[0]
    over = overlay_masks(image, lab); crop, box = crop_for_foreground(over, lab)
    save(out / "img_a2_sam2_mask_overlay.png", crop)
    flat = PALETTE[lab % len(PALETTE)]; flat_crop = flat[box[1]:box[3], box[0]:box[2]]
    save(out / "img_a3_label_image.png", flat_crop)
    hull_figure(root, out / "img_a4_hull_comparison.png")
    crop_matches(root, images, labs, out / "img_a5_crop_mast3r_matches.png")
    # eval image is a 2x2 GT/render diagnostic; upper-right is the Stage-A held-out render.
    panel = read_rgb(root / "eval" / "view_13_heldout.jpg")
    save(out / "img_a6_canonical_heldout_render.png", panel[:panel.shape[0] // 2, panel.shape[1] // 2:])
    controls_figure(root, out / "img_b0_controls_by_group.png")
    prompt_figure(root, out / "img_b2_prompt_boxes_and_masks.png")
    render = video_frame(root / "stage_b_diag_crop_ball" / "videos" / "heldout_view13.mp4", 150)
    save(out / "img_b5_composited_render_f150.png", render)
    print("Wrote", out)


if __name__ == "__main__":
    main()
