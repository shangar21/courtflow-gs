"""Per-frame images for Stage B, reproducing the frame-0 preprocessing exactly:
4K video -> ffmpeg lanczos to the capture scale (rgb24) -> OPENCV undistortion to the COLMAP pinhole camera
(bilinear, COLMAP pixel-centre convention) -> fixed per-view linear-sRGB gain/bias.

The undistortion and colour table can run on the GPU (``to_device``); training frames can be
stored as JPEG, which is an order of magnitude faster to write than PNG at 4K."""
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


def colmap_camera_id(images_txt: Path, view: int) -> int:
    """CAMERA_ID for ``view_NNN`` in a COLMAP images.txt, whatever the image extension.

    The raw capture names images ``view_NNN.jpg``; converted models use ``.png``."""
    stem = f"view_{view:03d}"
    for l in Path(images_txt).read_text().splitlines():
        parts = l.split()
        if l and not l.startswith("#") and len(parts) >= 10 and Path(parts[9]).stem == stem:
            return int(parts[8])
    raise KeyError(f"{stem} not found in {images_txt}")


def frame_file(directory: Path, frame: int) -> Path:
    """An extracted frame: JPEG (fast capture path) when present, otherwise lossless PNG."""
    jpg = Path(directory) / f"{frame:06d}.jpg"
    return jpg if jpg.is_file() else Path(directory) / f"{frame:06d}.png"


class ViewPreprocessor:
    def __init__(self, view: int, distorted_cameras_txt: Path, distorted_images_txt: Path, pinhole_sparse: Path, photometric_json: Path):
        pinhole_sparse = Path(pinhole_sparse)
        name = f"view_{view:03d}.png"
        cam_id = colmap_camera_id(distorted_images_txt, view)
        line = next(l.split() for l in Path(distorted_cameras_txt).read_text().splitlines() if l and not l.startswith("#") and int(l.split()[0]) == cam_id)
        if line[1] != "OPENCV": raise ValueError(f"{name}: expected OPENCV camera model, got {line[1]}")
        self.src_size = (int(line[2]), int(line[3]))
        fx, fy, cx, cy, k1, k2, p1, p2 = map(float, line[4:12])
        if pinhole_sparse.suffix == ".json":
            item = next(x for x in json.loads(pinhole_sparse.read_text())["cameras"] if x.get("name") == name)
            Kout, out_w, out_h = np.asarray(item["K"], np.float32), int(item["width"]), int(item["height"])
            # The capture bootstrap decodes raw videos at the pinhole cameras' scale (capture_scale).
            k = out_w / self.src_size[0]
            self.src_size = (out_w, round(self.src_size[1] * k)); fx *= k; fy *= k; cx *= k; cy *= k
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
        self._grid = self._lut = None

    def to_device(self, device: str) -> "ViewPreprocessor":
        """Run undistortion (same map, bilinear) and the colour table on ``device``."""
        import torch
        w, h = self.src_size
        gx = torch.as_tensor(self.map1, dtype=torch.float32) / (w - 1) * 2 - 1
        gy = torch.as_tensor(self.map2, dtype=torch.float32) / (h - 1) * 2 - 1
        self._grid = torch.stack((gx, gy), -1)[None].to(device)
        self._lut = torch.as_tensor(np.stack(self.lut), dtype=torch.long, device=device)
        return self

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self._grid is not None:
            return self._gpu(rgb)
        u = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR)
        return np.stack([self.lut[c][u[..., c]] for c in range(3)], -1)

    def _gpu(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        rgb = np.require(rgb, requirements=("C", "W"))   # ffmpeg buffers are read-only; torch needs writable memory
        x = torch.from_numpy(rgb).to(self._grid.device).permute(2, 0, 1)[None].float()
        u = F.grid_sample(x, self._grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        u = u.round_().clamp_(0, 255).long().flatten(1)
        out = self._lut.gather(1, u).view(3, *self._grid.shape[1:3]).permute(1, 2, 0)
        return out.to(torch.uint8).cpu().numpy()


def decode_frames(video: Path, start: int, end: int, width: int, height: int, stride: int = 1):
    """Yields (frame_index, RGB uint8 at width x height) for every ``stride``-th frame of
    [start, end), using the same ffmpeg resampling as frame 0. Skipped frames are still decoded
    (HEVC needs them) but are dropped before the resize and RGB conversion."""
    keep = f"select='not(mod(n\\,{stride}))'," if stride > 1 else ""
    vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,{keep}scale={width}:{height}:flags=lanczos"
    frames = range(start, end, stride)
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf, "-vsync", "0", "-frames:v", str(len(frames)),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE); size = width * height * 3
    try:
        for f in frames:
            buf = proc.stdout.read(size)
            if len(buf) < size: raise RuntimeError(f"{video}: stream ended before frame {f}")
            yield f, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close(); proc.wait()


def extract_view(pre: ViewPreprocessor, video: Path, out_dir: Path, start: int, end: int, stride: int = 1,
                 ext: str = "png", quality: int = 95) -> int:
    """Decode, preprocess, and store every ``stride``-th frame as ``NNNNNN.<ext>`` (png | jpg)."""
    out_dir.mkdir(parents=True, exist_ok=True); n = 0
    if all((out_dir / f"{f:06d}.{ext}").is_file() for f in range(start, end, stride)): return 0   # nothing to decode
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    w, h = pre.src_size
    for f, rgb in decode_frames(video, start, end, w, h, stride):
        target = out_dir / f"{f:06d}.{ext}"
        if target.is_file(): continue
        cv2.imwrite(str(target), cv2.cvtColor(pre(rgb), cv2.COLOR_RGB2BGR), params); n += 1
    return n
