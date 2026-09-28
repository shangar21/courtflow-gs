import json, statistics as st
B = "/home/ubuntu/outputs/courtflow/basketball"
runs = {"uncapped": "stage_b_v2_uncapped", "1.1M": "stage_b_v2", "2M": "stage_b_cap2m", "3M": "stage_b_cap3m"}
for name, tag in runs.items():
    m = json.load(open(f"{B}/{tag}/metrics.json"))["frames"]; ev = [x for x in m if "eval" in x and "heldout_psnr" in x["eval"]]
    g = lambda k, lo=0, hi=700: st.mean(x["eval"][k] for x in ev if lo <= x["frame"] < hi)
    s = json.load(open(f"{B}/{tag}/summary.json"))
    print(f"{name:9s} n={len(ev)} full {g('heldout_psnr'):.2f} (1st {g('heldout_psnr',0,350):.2f} 2nd {g('heldout_psnr',350,700):.2f}) "
          f"player {g('heldout_person_psnr'):.2f} (1st {g('heldout_person_psnr',0,350):.2f} 2nd {g('heldout_person_psnr',350,700):.2f}) "
          f"ssim {g('heldout_ssim'):.3f} lpips {g('heldout_lpips'):.3f} iou {g('train_iou'):.3f} | {s['final_gaussians']/1e6:.2f}M {s['seconds_per_frame']:.2f}s/f kf {s['keyframe_s']:.1f}s reinit {sum(len(x.get('reinit',[])) for x in m)}")
