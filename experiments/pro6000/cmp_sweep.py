import json
B = "/home/ubuntu/outputs/courtflow/basketball"
runs = {"baseline": "stage_b_v2", "A kf-dyn+cap": "sweep_A", "B cap only": "sweep_B", "C A+refine-dyn": "sweep_C"}
data = {k: json.load(open(f"{B}/{v}/metrics.json"))["frames"] for k, v in runs.items()}
print("frame | " + " | ".join(f"{k:>22s}" for k in runs))
for f in (50, 100, 150, 200, 250, 290, 299):
    row = []
    for k, m in data.items():
        x = next((x for x in m if x["frame"] == f), None)
        if x is None or "eval" not in x: row.append(" " * 22); continue
        e = x["eval"]
        row.append(f'{e.get("heldout_psnr", float("nan")):5.2f} {e.get("heldout_person_psnr", float("nan")):5.2f} {e.get("train_iou", float("nan")):.2f} {x["gaussians"]/1e6:4.2f}M')
    print(f"{f:5d} | " + " | ".join(row))
for k, m in data.items():
    m = [x for x in m if x["frame"] <= 299]
    ev = [x["eval"] for x in m if "eval" in x and "heldout_psnr" in x["eval"]]
    big = [(x["frame"], round(x["first_loss"], 2)) for x in m if x.get("first_loss", 0) > 2]
    print(f"{k:16s} mean held {sum(e['heldout_psnr'] for e in ev)/len(ev):.2f}  person {sum(e['heldout_person_psnr'] for e in ev)/len(ev):.2f}  reinit {sum(len(x.get('reinit', [])) for x in m)}  loss>2 frames {big[:6]}")
