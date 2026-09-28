"""Resumed run (from rt_full:20) must reproduce the uninterrupted run's schedule and outputs."""
import json, subprocess, numpy as np
B = "/home/ubuntu/outputs/courtflow/basketball"
m = {t: json.load(open(f"{B}/{t}/metrics.json"))["frames"] for t in ("rt_full", "rt_res")}
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, detail)
check("same frame list", [x["frame"] for x in m["rt_full"]] == [x["frame"] for x in m["rt_res"]], f'{len(m["rt_res"])} frames')
kf = {t: [x["frame"] for x in v if "keyframe_s" in x] for t, v in m.items()}
check("same keyframes", kf["rt_full"] == kf["rt_res"], str(kf["rt_res"]))
mk = {t: [x["frame"] for x in v if "reassoc" in x] for t, v in m.items()}
check("same mask frames", mk["rt_full"] == mk["rt_res"], str(mk["rt_res"]))
check("frames <=20 copied exactly", all(a == b for a, b in zip(m["rt_full"][:20], m["rt_res"][:20])))
g = {t: [x["gaussians"] for x in v] for t, v in m.items()}
check("absolute cap respected", max(g["rt_full"] + g["rt_res"]) <= 800_000, f'max {max(g["rt_res"])}')
for t in ("rt_full", "rt_res"):
    P = np.load(f"{B}/{t}/control_positions.npz")["pos"]; check(f"{t} control paths cover 60 frames", len(P) == 60, str(P.shape))
    for v in ("heldout_view13.mp4", "orbit360.mp4"):
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", f"{B}/{t}/videos/{v}"], capture_output=True, text=True).stdout.strip()
        check(f"{t} {v} has 60 frames", n == "60", n)
# first 21 video frames of the resumed run are the source's frames
a = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{B}/rt_full/videos/orbit360.mp4", "-frames:v", "21", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
b = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{B}/rt_res/videos/orbit360.mp4", "-frames:v", "21", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
d = np.abs(np.frombuffer(a, np.uint8).astype(int) - np.frombuffer(b, np.uint8).astype(int)).mean() if len(a) == len(b) else 999
check("copied video frames match source (re-encode only)", d < 2.0, f"mean abs diff {d:.2f}")
ev = {t: {x["frame"]: x["eval"] for x in v if "eval" in x and "heldout_psnr" in x["eval"]} for t, v in m.items()}
for f in (30, 40, 50, 59):
    if f in ev["rt_full"] and f in ev["rt_res"]:
        a, b = ev["rt_full"][f], ev["rt_res"][f]
        check(f"frame {f} test PSNR close", abs(a["heldout_psnr"] - b["heldout_psnr"]) < 0.3, f'{a["heldout_psnr"]:.2f} vs {b["heldout_psnr"]:.2f}; player {a["heldout_person_psnr"]:.2f} vs {b["heldout_person_psnr"]:.2f}; IoU {a["train_iou"]:.3f} vs {b["train_iou"]:.3f}')
print("ALL PASS" if ok else "SOME FAILED")
