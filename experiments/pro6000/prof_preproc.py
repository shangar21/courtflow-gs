"""Split Stage-B frame preprocessing cost: decode (CPU vs NVDEC), undistort+LUT (CPU vs GPU), PNG write."""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from ring_init.io.frames import ViewPreprocessor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cfg = json.load(open("/home/ubuntu/outputs/courtflow/basketball/e2e_config.json"))
video = Path(cfg["videos_dir"]) / "view_000.mp4"
pre = ViewPreprocessor(0, Path(cfg["distorted_cameras_txt"]), Path(cfg["distorted_images_txt"]), Path(cfg["eval_sparse"]), Path(cfg["photometric_json"]))
res = {}

def decode(w, h, gpu):
    if gpu:
        vf = f"scale_cuda={w}:{h}:interp_algo=lanczos:format=yuv420p,hwdownload,format=yuv420p"
        cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "hevc_cuvid", "-i", str(video),
               "-frames:v", str(N), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", str(N), "-vf", f"scale={w}:{h}:flags=lanczos",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    t = time.perf_counter(); p = subprocess.Popen(cmd, stdout=subprocess.PIPE); frames = []
    while len(frames) < 3:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3: break
        frames.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
    while p.stdout.read(1 << 24): pass
    p.wait(); dt = (time.perf_counter() - t) / N
    if p.returncode: raise RuntimeError(" ".join(cmd))
    return dt, frames[0]

for label, (w, h) in (("1080p", (1920, 1080)), ("4K", (3840, 2160))):
    for gpu in (False, True):
        try:
            dt, fr = decode(w, h, gpu); res[f"{label} decode {'NVDEC' if gpu else 'CPU'} ms"] = dt * 1e3
        except Exception as e:
            res[f"{label} decode {'NVDEC' if gpu else 'CPU'} ms"] = f"FAILED {e}"
    # CPU undistort + LUT at this size (maps rebuilt at this scale for 4K)
    k = w / pre.map1.shape[1]
    m1, m2 = (pre.map1, pre.map2) if k == 1 else (cv2.resize(pre.map1, (w, h)) * k, cv2.resize(pre.map2, (w, h)) * k)
    img = np.ascontiguousarray(fr); t = time.perf_counter()
    for _ in range(20):
        u = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
        u = np.stack([pre.lut[c][u[..., c]] for c in range(3)], -1)
    res[f"{label} undistort+LUT CPU ms"] = (time.perf_counter() - t) / 20 * 1e3
    # GPU equivalent: grid_sample with the same map + LUT gather
    grid = torch.stack([torch.as_tensor(m1) / (w - 1) * 2 - 1, torch.as_tensor(m2) / (h - 1) * 2 - 1], -1)[None].cuda()
    lut = torch.as_tensor(np.stack(pre.lut)).cuda().long()
    x = torch.as_tensor(img).cuda()
    for it in range(25):
        if it == 5: torch.cuda.synchronize(); t = time.perf_counter()
        g = F.grid_sample(x.permute(2, 0, 1)[None].float(), grid, mode="bilinear", align_corners=True)[0].round().clamp(0, 255).long()
        g = torch.stack([lut[c][g[c]] for c in range(3)], -1).to(torch.uint8)
    torch.cuda.synchronize(); res[f"{label} undistort+LUT GPU ms"] = (time.perf_counter() - t) / 20 * 1e3
    d = Path(tempfile.mkdtemp()); t = time.perf_counter()
    for i in range(10): cv2.imwrite(str(d / f"{i}.png"), u, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    res[f"{label} PNG write ms"] = (time.perf_counter() - t) / 10 * 1e3
    res[f"{label} PNG MB"] = (d / "0.png").stat().st_size / 1e6
    t = time.perf_counter()
    for i in range(10): cv2.imwrite(str(d / f"{i}.jpg"), u, [cv2.IMWRITE_JPEG_QUALITY, 95])
    res[f"{label} JPEG95 write ms"] = (time.perf_counter() - t) / 10 * 1e3
    t = time.perf_counter()
    for i in range(10): cv2.imread(str(d / f"{i}.png"))
    res[f"{label} PNG read ms"] = (time.perf_counter() - t) / 10 * 1e3
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
