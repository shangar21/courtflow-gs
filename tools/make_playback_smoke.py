"""Create a tiny CUDA-playback package for an immediate local smoke test.

Run:
  python tools/make_playback_smoke.py /tmp/courtflow-playback-smoke
  python -m ring_init.play --package /tmp/courtflow-playback-smoke --camera view:0 --out /tmp/smoke.mp4
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np


def save(path: Path, means: np.ndarray, ids: np.ndarray) -> None:
    n = len(means)
    np.savez_compressed(path, means=means.astype(np.float16),
                        quats=np.tile(np.array([1, 0, 0, 0], np.float16), (n, 1)),
                        scales=np.full((n, 3), -2.0, np.float16),
                        opacities=np.full(n, 4.0, np.float16),
                        sh0=np.full((n, 1, 3), 1.5, np.float16),
                        sh1=np.zeros((n, 3, 3), np.float16),
                        instance_ids=ids.astype(np.int16))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) == 2 else "/tmp/courtflow-playback-smoke")
    root.mkdir(parents=True, exist_ok=True)
    meta = {"format_version": 1, "fps": 25, "start_frame": 0, "end_frame": 2,
            "background_keyframe_every": 10,
            "cameras": [{"view": 0, "K": [[320, 0, 160], [0, 320, 120], [0, 0, 1]],
                         "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0],
                         "width": 320, "height": 240}]}
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    save(root / "background_000000.npz", np.array([[0, 0, 4.0], [-.4, 0, 4.5], [.4, 0, 4.5]]), np.zeros(3))
    for frame in range(3):
        save(root / f"players_{frame:06d}.npz", np.array([[.15 * frame - .15, .1, 3.5]]), np.ones(1))
    print(root)


if __name__ == "__main__":
    main()
