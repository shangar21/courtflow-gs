"""Per-camera person masks: pluggable box prompts (detector by default) refined by SAM2."""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import torch

BoxPrompter = Callable[[np.ndarray], np.ndarray]  # RGB image -> (N, 4) xyxy boxes


def torchvision_person_detector(name: str, min_size: int, max_size: int, score_threshold: float, max_detections: int, device: str = "cuda") -> BoxPrompter:
    from torchvision.models import detection
    builders = {"fasterrcnn_resnet50_fpn_v2": (detection.fasterrcnn_resnet50_fpn_v2, detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
                "fasterrcnn_resnet50_fpn": (detection.fasterrcnn_resnet50_fpn, detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)}
    if name not in builders: raise ValueError(f"Unknown detector {name!r}; choose from {sorted(builders)}")
    build, weights = builders[name]
    model = build(weights=weights, box_detections_per_img=max_detections).to(device).eval()
    model.transform.min_size = (min_size,); model.transform.max_size = max_size

    @torch.inference_mode()
    def prompt(rgb: np.ndarray) -> np.ndarray:
        out = model([torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= score_threshold)
        return out["boxes"][keep].cpu().numpy()
    return prompt


class SAM2Refiner:
    def __init__(self, checkpoint: str | None, model_cfg: str | None, device: str = "cuda"):
        if not checkpoint or not model_cfg:
            raise RuntimeError("SAM2 requires config.sam2_checkpoint and config.sam2_config.")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError("SAM2 is not installed (github.com/facebookresearch/sam2).") from error
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

    def set_image(self, rgb: np.ndarray) -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb)

    def masks(self, boxes: np.ndarray, points: np.ndarray | None = None, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Box (+ optional single positive point per box) prompts -> (N,H,W) bool masks, scores."""
        out_m, out_s = [], []
        for s in range(0, len(boxes), chunk):
            b = boxes[s:s+chunk]
            kw = {}
            if points is not None:
                kw = {"point_coords": points[s:s+chunk, None, :], "point_labels": np.ones((len(b), 1), np.int32)}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                m, sc, _ = self.predictor.predict(box=b, multimask_output=False, **kw)
            m = np.asarray(m).reshape(len(b), *m.shape[-2:]) > 0; sc = np.asarray(sc).reshape(len(b), -1)[:, 0]
            out_m.append(m); out_s.append(sc)
        if not out_m: return np.zeros((0, *self.predictor._orig_hw[0]), bool), np.zeros(0)
        return np.concatenate(out_m), np.concatenate(out_s)


def person_masks(images: list[np.ndarray], prompter: BoxPrompter, refiner: SAM2Refiner, min_area_px: int) -> list[dict]:
    """For every camera: boxes, SAM2 masks (bool N,H,W) and their union."""
    result = []
    for rgb in images:
        boxes = prompter(rgb)
        refiner.set_image(rgb)
        masks, scores = refiner.masks(boxes) if len(boxes) else (np.zeros((0, *rgb.shape[:2]), bool), np.zeros(0))
        keep = masks.reshape(len(masks), -1).sum(1) >= min_area_px
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
        union = masks.any(0) if len(masks) else np.zeros(rgb.shape[:2], bool)
        result.append({"boxes": boxes, "masks": masks, "scores": scores, "union": union})
    return result


def instance_histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [16, 16], [0, 180, 0, 256]).ravel()
    return hist / max(float(hist.sum()), 1.0)


def save_label_masks(labels: list[np.ndarray], out_dir: str | Path) -> None:
    """labels[c] is an int32 image (0 = background). Writes one PNG per camera per instance."""
    root = Path(out_dir)
    for cam, lab in enumerate(labels):
        d = root / f"cam_{cam:02d}"; d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("instance_*.png"): old.unlink()
        for ident in np.unique(lab):
            if ident == 0: continue
            cv2.imwrite(str(d / f"instance_{int(ident):03d}.png"), (lab == ident).astype(np.uint8)*255)


def load_label_masks(out_dir: str | Path, shapes: list[tuple[int, int]]) -> list[np.ndarray]:
    """shapes[c] = (height, width) of camera c."""
    labels = []
    for cam, shape in enumerate(shapes):
        lab = np.zeros(shape, np.int32)
        for p in sorted((Path(out_dir) / f"cam_{cam:02d}").glob("instance_*.png")):
            lab[cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0] = int(p.stem.split("_")[-1])
        labels.append(lab)
    return labels
