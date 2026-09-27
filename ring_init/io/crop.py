"""Pixel-box helpers shared by crop matching and crop rendering."""
from __future__ import annotations
import numpy as np


def square_box(lo: np.ndarray, hi: np.ndarray, padding_fraction: float, min_size: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Padded square box (x0, y0, x1, y1), shifted (not shrunk) to stay inside the image."""
    center = (np.asarray(lo, float) + np.asarray(hi, float)) / 2
    side = int(np.ceil(max(float(np.max(np.asarray(hi) - np.asarray(lo))) * (1 + 2*padding_fraction), min_size)))
    side = min(side, width, height)
    x0 = int(np.clip(round(center[0] - side/2), 0, width - side)); y0 = int(np.clip(round(center[1] - side/2), 0, height - side))
    return x0, y0, x0 + side, y0 + side


def to_crop(uv: np.ndarray, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Full-image pixel coordinates -> resized-crop pixel coordinates (pixel-centre convention)."""
    x0, y0, x1, _ = box; s = size / (x1 - x0)
    return (np.asarray(uv, float) - [x0, y0] + 0.5) * s - 0.5


def from_crop(uv: np.ndarray, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, y0, x1, _ = box; s = size / (x1 - x0)
    return (np.asarray(uv, float) + 0.5) / s - 0.5 + [x0, y0]
