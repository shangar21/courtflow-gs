from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class FrameMetrics:
    frame: int
    psnr: float
    ssim: float
    lpips: float
    alpha_iou: float
    seconds: float
    mask_propagation: float = 0.
    render: float = 0.
    mls_fwd: float = 0.
    mls_bwd: float = 0.
    optimizer: float = 0.


def write_temporal_report(rows: list[FrameMetrics], path: str | Path) -> None:
    values=[asdict(x) for x in rows]; drift=values[0]["psnr"]-values[-1]["psnr"] if len(values)>1 else 0.
    Path(path).write_text(json.dumps({"frames":values,"psnr_drift_first_minus_last":drift},indent=2)+"\n")
