"""Real-data check of the 4K fast frame path: bootstrap at capture_scale 1.0, then extract one
camera with the CPU/PNG path and the GPU/JPEG path; compare time and pixels."""
import time, shutil
from pathlib import Path
import cv2, numpy as np
import ring_init
from ring_init.bootstrap import prepare
from ring_init.io.frames import ViewPreprocessor, extract_view

print("ring_init from", ring_init.__file__)
t = time.time(); P = prepare("/home/ubuntu/datasets/basketball", "/home/ubuntu/outputs/courtflow_4k", "basketball", 1.0)
print(f"4K bootstrap {time.time() - t:.1f}s")
view, N = 0, 20
video = Path(P["videos_dir"]) / f"view_{view:03d}.mp4"
mk = lambda: ViewPreprocessor(view, Path(P["distorted_cameras_txt"]), Path(P["distorted_images_txt"]), Path(P["eval_sparse"]), Path(P["photometric_json"]))
root = Path("/home/ubuntu/outputs/fastframe_check"); shutil.rmtree(root, ignore_errors=True)
pre = mk(); print("src_size", pre.src_size)
t = time.time(); extract_view(pre, video, root / "cpu_png", 0, N, 1, "png"); cpu = (time.time() - t) / N
pre = mk().to_device("cuda")
t = time.time(); extract_view(pre, video, root / "gpu_jpg", 0, N, 1, "jpg", 95); gpu = (time.time() - t) / N
print(f"per frame: CPU+PNG {cpu*1e3:.0f} ms, GPU+JPEG {gpu*1e3:.0f} ms, speed-up {cpu/gpu:.1f}x")
ps = []
for f in range(N):
    a = cv2.imread(str(root / "cpu_png" / f"{f:06d}.png")).astype(np.float64); b = cv2.imread(str(root / "gpu_jpg" / f"{f:06d}.jpg")).astype(np.float64)
    ps.append(10 * np.log10(255 ** 2 / ((a - b) ** 2).mean()))
print(f"GPU+JPEG vs CPU+PNG PSNR: mean {np.mean(ps):.1f} dB, min {np.min(ps):.1f} dB; shape {a.shape}")
# Frame 0 must agree with the bootstrap's own frame-0 image (Stage A's training target).
ref = cv2.imread(str(Path(P["eval_images"]) / f"view_{view:03d}.png")).astype(np.float64)
a0 = cv2.imread(str(root / "cpu_png" / "000000.png")).astype(np.float64)
print(f"frame 0 vs bootstrap image PSNR: CPU path {10*np.log10(255**2/((a0-ref)**2).mean()):.1f} dB")
print("sizes MB: png", round((root / "cpu_png" / "000005.png").stat().st_size / 1e6, 2), "jpg", round((root / "gpu_jpg" / "000005.jpg").stat().st_size / 1e6, 2))
