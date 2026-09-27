"""MASt3R dense matching (no pose estimation) on full frames or on per-instance crops."""
from __future__ import annotations
import cv2
import numpy as np
import torch


def ring_pairs(camera_count: int = 12) -> list[tuple[int, int]]:
    """Neighbour (i, i+1) and skip-one (i, i+2) pairs around the ring: 24 for 12 cameras."""
    return [(i, (i+d) % camera_count) for d in (1, 2) for i in range(camera_count)]


class MASt3RMatcher:
    """One model instance reused for every pair. Images are passed as RGB uint8 arrays whose
    sides are multiples of 16; returned pixels are in those arrays' coordinates."""

    def __init__(self, model_name: str, device: str):
        try:
            from mast3r.model import AsymmetricMASt3R
        except ImportError as error:
            raise RuntimeError("MASt3R must be importable: PYTHONPATH=third_party/MAtCha/mast3r:third_party/MAtCha/mast3r/dust3r") from error
        self.device = device
        self.model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()

    def _view(self, rgb: np.ndarray, idx: int) -> dict:
        h, w = rgb.shape[:2]
        if h % 16 or w % 16: raise ValueError(f"MASt3R input sides must be multiples of 16, got {w}x{h}")
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1)[None]
        return {"img": tensor, "true_shape": np.array([[h, w]], np.int32), "idx": idx, "instance": str(idx)}

    @torch.no_grad()
    def match(self, rgb_a: np.ndarray, rgb_b: np.ndarray, max_matches: int, seeds: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reciprocal descriptor NN matches. `seeds` = (x, y) query pixels in image a (e.g. only an
        instance mask); otherwise a regular grid sized to ~max_matches."""
        from dust3r.inference import inference
        from mast3r.fast_nn import fast_reciprocal_NNs
        result = inference([(self._view(rgb_a, 0), self._view(rgb_b, 1))], self.model, self.device, batch_size=1, verbose=False)
        p1, p2 = result["pred1"], result["pred2"]
        d1, d2 = p1["desc"].squeeze(0).detach(), p2["desc"].squeeze(0).detach()
        h, w = rgb_a.shape[:2]
        init = seeds if seeds is not None else max(1, int(np.sqrt(h*w / max_matches)))
        if seeds is not None and len(seeds[0]) == 0: return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, np.float32)
        uv1, uv2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=init, device=self.device, dist="dot", block_size=2**13)
        uv1, uv2 = np.asarray(uv1), np.asarray(uv2)
        c1 = p1["desc_conf"].squeeze(0).cpu().numpy(); c2 = p2["desc_conf"].squeeze(0).cpu().numpy()
        conf = np.minimum(c1[uv1[:, 1], uv1[:, 0]], c2[uv2[:, 1], uv2[:, 0]])
        return uv1.astype(np.float64), uv2.astype(np.float64), conf.astype(np.float32)


def resize_full(rgb: np.ndarray, long_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize a full frame to multiple-of-16 sides; returns image and per-axis scale (full/resized)."""
    h, w = rgb.shape[:2]; s = long_side / max(h, w)
    nw, nh = max(16, int(round(w*s/16))*16), max(16, int(round(h*s/16))*16)
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA), np.array([w/nw, h/nh])


def resized_to_full(uv: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(uv, float) + 0.5) * scale - 0.5
