"""Instance masks through time: SAM2 video predictor per training camera, prompted with the
Stage A frame-0 instance masks. Propagation is causal (frame f uses frames <= f), so running it
over a buffered clip is equivalent to online use; per-frame cost is logged."""
from __future__ import annotations
import shutil
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import torch


def propagate_camera(predictor, png_dir: Path, frames: list[int], frame0_labels: np.ndarray, out_dir: Path, min_area_px: int) -> dict:
    """Writes out_dir/NNNNNN.png int label images (0 = background) for every frame in `frames`
    (frames[0] must be the prompted frame). Overlaps go to the object with the highest logit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(k) for k in np.unique(frame0_labels) if k > 0]
    tmp = Path(tempfile.mkdtemp(prefix="sam2_jpg_"))
    try:
        for i, f in enumerate(frames):  # SAM2 reads "<index>.jpg" folders
            im = cv2.imread(str(png_dir / f"{f:06d}.png"))
            if im is None: raise FileNotFoundError(png_dir / f"{f:06d}.png")
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(tmp), offload_video_to_cpu=True, async_loading_frames=False)
            load_s = time.time() - t0
            for k in ids: predictor.add_new_mask(state, 0, k, frame0_labels == k)
            t1 = time.time(); written = 0
            for idx, obj_ids, logits in predictor.propagate_in_video(state):
                logits = logits[:, 0].float()                       # [O,H,W]
                best = logits.max(0)
                lab = torch.where(best.values > 0, torch.as_tensor(obj_ids, device=logits.device)[best.indices], 0).cpu().numpy().astype(np.uint8)
                for k in obj_ids:
                    if 0 < (lab == k).sum() < min_area_px: lab[lab == k] = 0
                cv2.imwrite(str(out_dir / f"{frames[idx]:06d}.png"), lab); written += 1
            prop_s = time.time() - t1
        predictor.reset_state(state); del state; torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"frames": written, "objects": ids, "load_s": load_s, "propagate_s": prop_s, "propagate_s_per_frame": prop_s / max(written, 1)}


def load_video_labels(out_dir: Path, frame: int) -> np.ndarray:
    lab = cv2.imread(str(out_dir / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if lab is None: raise FileNotFoundError(out_dir / f"{frame:06d}.png")
    return lab.astype(np.int32)
