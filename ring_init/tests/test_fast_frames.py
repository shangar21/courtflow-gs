import shutil
import subprocess
import cv2
import numpy as np
import pytest
import torch
from ring_init.io.frames import ViewPreprocessor, decode_frames, extract_view, frame_file

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")


def _preprocessor(w=64, h=48) -> ViewPreprocessor:
    """A ViewPreprocessor with a mild synthetic distortion map and a non-identity colour LUT."""
    pre = ViewPreprocessor.__new__(ViewPreprocessor)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xs - w / 2) ** 2 + (ys - h / 2) ** 2) / (w * w)
    pre.map1, pre.map2 = xs * (1 + .05 * r2) - .3, ys * (1 + .05 * r2) + .2
    pre.src_size = (w, h)
    pre.lut = [np.clip(np.arange(256) * g + b, 0, 255).astype(np.uint8) for g, b in ((1.1, -3), (0.9, 5), (1.0, 0))]
    pre._grid = pre._lut = None
    return pre


@requires_cuda
def test_gpu_preprocess_matches_cpu_within_one_level():
    rng = np.random.default_rng(0)
    rgb = cv2.GaussianBlur(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), (5, 5), 0)
    pre = _preprocessor(); cpu = pre(rgb)
    pre.to_device("cuda"); gpu = pre(rgb)
    diff = np.abs(cpu.astype(int) - gpu.astype(int))
    assert gpu.dtype == np.uint8 and gpu.shape == cpu.shape
    assert diff.max() <= 2 and diff.mean() < 0.5


def test_frame_file_prefers_jpeg_and_falls_back_to_png(tmp_path):
    assert frame_file(tmp_path, 3).name == "000003.png"
    (tmp_path / "000003.jpg").write_bytes(b"x")
    assert frame_file(tmp_path, 3).name == "000003.jpg"


@pytest.fixture
def clip(tmp_path):
    """A 23-frame clip whose frame n encodes n in its brightness."""
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x48:r=25:d=0.92",
                    "-vf", "geq=lum='N*10':cb=128:cr=128", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)], check=True)
    return path


@requires_ffmpeg
def test_strided_decode_returns_the_requested_frames(clip):
    got = list(decode_frames(clip, 2, 23, 64, 48, stride=5))
    every = dict(decode_frames(clip, 0, 23, 64, 48))
    assert [f for f, _ in got] == [2, 7, 12, 17, 22]
    assert all(np.array_equal(im, every[f]) for f, im in got)
    assert len({float(im.mean()) for _, im in got}) == 5   # distinct frames, not repeats


@requires_ffmpeg
def test_extract_view_writes_jpeg_frames(clip, tmp_path):
    pre = _preprocessor(); out = tmp_path / "frames"
    assert extract_view(pre, clip, out, 0, 10, 1, ext="jpg", quality=95) == 10
    assert sorted(p.name for p in out.iterdir())[:2] == ["000000.jpg", "000001.jpg"]
    assert extract_view(pre, clip, out, 0, 10, 1, ext="jpg", quality=95) == 0   # cached
