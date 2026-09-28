"""Compact, self-contained export format for :mod:`ring_init.play`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PlaybackExporter:
    """Write exactly the Gaussian fields needed by the forward-only player.

    Background parameters are saved at keyframes; player/ball parameters are saved for every
    frame.  All continuous parameters use float16 on disk, while instance IDs remain integers.
    """

    def __init__(self, root: Path, cameras: dict, fps: float, start: int, end: int, keyframe_every: int):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "fps": fps,
            "start_frame": start,
            "end_frame": end - 1,
            "background_keyframe_every": keyframe_every,
            "cameras": [
                {"view": int(view), "K": np.asarray(cam.K).tolist(), "R": np.asarray(cam.R).tolist(), "t": np.asarray(cam.t).tolist(),
                 "width": int(cam.width), "height": int(cam.height)}
                for view, cam in sorted(cameras.items())
            ],
        }
        (root / "meta.json").write_text(json.dumps(payload, indent=2) + "\n")

    @staticmethod
    def _save(path: Path, params: dict, mask) -> None:
        fields = ("means", "quats", "scales", "opacities", "sh0", "shN")
        out = {key: params[key].detach()[mask].cpu().numpy().astype(np.float16) for key in fields}
        out["instance_ids"] = params["instance_ids"].detach()[mask].cpu().numpy().astype(np.int16)
        # play.py uses the first three degree-1 coefficients; retain that explicit compact form.
        out["sh1"] = out.pop("shN")[:, :3]
        np.savez_compressed(path, **out)

    def write_frame(self, frame: int, model, write_background: bool) -> None:
        params = model.params
        ids = params["instance_ids"].detach()
        self._save(self.root / f"players_{frame:06d}.npz", params, ids != 0)
        if write_background:
            self._save(self.root / f"background_{frame:06d}.npz", params, ids == 0)
