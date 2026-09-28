"""Parallel SAM2 masks must equal the serial masks, camera by camera."""
import shutil, time
from pathlib import Path
import cv2, numpy as np
from ring_init.config import Config
from ring_init.stage_a import Scene
from ring_init.stage_b import step_video_masks

if __name__ == "__main__":
    cfg = Config.load("/home/ubuntu/outputs/courtflow_smoke/basketball/e2e_config.json"); cfg.sam2_workers = 3
    s = Scene("basketball", cfg, set()); out = s.out / "video_masks_par"; shutil.rmtree(out, ignore_errors=True)
    t = time.time(); step_video_masks(s, 0, 30, name="video_masks_par"); print(f"parallel: {time.time() - t:.1f}s")
    diffs = []
    for ci in range(12):
        for f in range(30):
            a = cv2.imread(str(s.out / "video_masks" / f"cam_{ci:02d}" / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(out / f"cam_{ci:02d}" / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
            diffs.append(float((a != b).mean()))
    print(f"label images compared: {len(diffs)}, identical: {sum(d == 0 for d in diffs)}, max differing pixel fraction {max(diffs):.2e}")
    shutil.rmtree(out, ignore_errors=True)
