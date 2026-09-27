"""Per-frame images for Stage B, reproducing the frame-0 preprocessing exactly:
4K video -> ffmpeg lanczos 1/2 (rgb24) -> OPENCV undistortion to the COLMAP pinhole camera
(bilinear, COLMAP pixel-centre convention) -> fixed per-view linear-sRGB gain/bias."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, None), 1 / 2.4) - .055)


class ViewPreprocessor:
    def __init__(self, view: int, distorted_cameras_txt: Path, distorted_images_txt: Path, pinhole_sparse: Path, photometric_json: Path):
        pinhole_sparse = Path(pinhole_sparse)
        name = f"view_{view:03d}.png"
        cam_id = next(int(l.split()[8]) for l in Path(distorted_images_txt).read_text().splitlines()
                      if l and not l.startswith("#") and len(l.split()) >= 10 and l.split()[9] == name)
        line = next(l.split() for l in Path(distorted_cameras_txt).read_text().splitlines() if l and not l.startswith("#") and int(l.split()[0]) == cam_id)
        if line[1] != "OPENCV": raise ValueError(f"{name}: expected OPENCV camera model, got {line[1]}")
        self.src_size = (int(line[2]), int(line[3]))
        fx, fy, cx, cy, k1, k2, p1, p2 = map(float, line[4:12])
        if pinhole_sparse.suffix == ".json":
            item = next(x for x in json.loads(pinhole_sparse.read_text())["cameras"] if x.get("name") == name)
            Kout, out_w, out_h = np.asarray(item["K"], np.float32), int(item["width"]), int(item["height"])
            # Raw videos are decoded at half resolution by the capture bootstrap.
            self.src_size = (self.src_size[0] // 2, self.src_size[1] // 2); fx *= .5; fy *= .5; cx *= .5; cy *= .5
        else:
            import pycolmap
            rec = pycolmap.Reconstruction(str(pinhole_sparse)); im = next(i for i in rec.images.values() if i.name == name); c = rec.camera(im.camera_id)
            P, out_w, out_h = c.params, c.width, c.height
            Kout = np.array([[P[0], 0, P[2]], [0, P[1], P[3]], [0, 0, 1]], np.float32)
        # COLMAP pixel centres are at +0.5; OpenCV's at 0.
        Kd = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1]]); Ku = Kout.copy(); Ku[:2, 2] -= .5
        self.map1, self.map2 = cv2.initUndistortRectifyMap(Kd, np.array([k1, k2, p1, p2]), None, Ku, (out_w, out_h), cv2.CV_32FC1)
        photo = json.loads(Path(photometric_json).read_text())["transforms_to_reference"]
        t = next(x for x in photo if x["view"] == view)
        self.gain, self.bias = np.asarray(t["gain"], np.float32), np.asarray(t["bias"], np.float32)
        # 256-entry sRGB->linear table and a fine linear->sRGB table make the per-pixel mapping cheap.
        self.lut = [np.clip(_linear_to_srgb(np.clip(_srgb_to_linear(np.arange(256) / 255.0) * g + b, 0, 1)) * 255 + .5, 0, 255).astype(np.uint8)
                    for g, b in zip(self.gain, self.bias)]
        self.view = view; self.name = name

    def __call__(self, rgb_1080: np.ndarray) -> np.ndarray:
        u = cv2.remap(rgb_1080, self.map1, self.map2, cv2.INTER_LINEAR)
        return np.stack([self.lut[c][u[..., c]] for c in range(3)], -1)


def decode_frames(video: Path, start: int, end: int, width: int, height: int):
    """Yields (frame_index, RGB uint8 at width x height) using the same ffmpeg filter as frame 0."""
    vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos"
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf, "-frames:v", str(end - start), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE); size = width * height * 3
    try:
        for f in range(start, end):
            buf = proc.stdout.read(size)
            if len(buf) < size: raise RuntimeError(f"{video}: stream ended before frame {f}")
            yield f, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close(); proc.wait()


def extract_view(pre: ViewPreprocessor, video: Path, out_dir: Path, start: int, end: int, stride: int = 1) -> int:
    out_dir.mkdir(parents=True, exist_ok=True); n = 0
    if all((out_dir / f"{f:06d}.png").is_file() for f in range(start, end, stride)): return 0   # nothing to decode
    w, h = pre.src_size
    for f, rgb in decode_frames(video, start, end, w, h):
        if (f - start) % stride: continue
        target = out_dir / f"{f:06d}.png"
        if target.is_file(): continue
        cv2.imwrite(str(target), cv2.cvtColor(pre(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
    return n
