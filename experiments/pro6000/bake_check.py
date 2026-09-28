"""Does baking checkpoint 600 with control positions reproduce each player's position at frame 650?"""
import numpy as np, torch, torch.nn.functional as F
from ring_init.deform import mls
run = "/home/ubuntu/outputs/courtflow/basketball/stage_b_cap2m"
a = torch.load(f"{run}/state/frame_000600.pt", map_location="cuda", weights_only=False)["tracker"]
b = torch.load(f"{run}/state/frame_000650.pt", map_location="cuda", weights_only=False)["tracker"]
P = torch.as_tensor(np.load(f"{run}/control_positions.npz")["pos"], device="cuda")
pa, pb = a["params"], b["params"]
print("ctrl pos in checkpoint 650 == npz[650]:", float((b["ctrl"]["pos"] - P[650]).abs().max()), " ckpt600 vs npz[600]:", float((a["ctrl"]["pos"] - P[600]).abs().max()))
sh1 = pa["shN"][:, :3]
m, q, sh = mls.deform(pa["means"], F.normalize(pa["quats"], dim=-1), sh1, a["ctrl"]["pos"], a["ctrl"]["nbr"], a["ctrl"]["w"], (P[650] - P[600]).float().contiguous())
ia, ib = pa["instance_ids"].long(), pb["instance_ids"].long()
print("id  n600    n650   moved(600->650,m)  baked-vs-actual centroid err  (units)")
for k in sorted(set(ia.unique().tolist()) - {0}):
    ca, cb, cm = pa["means"][ia == k].mean(0), pb["means"][ib == k].mean(0), m[ia == k].mean(0)
    print(f"{k:3d} {int((ia==k).sum()):6d} {int((ib==k).sum()):6d}  moved {float((cb-ca).norm()):.3f}  baked err {float((cm-cb).norm()):.3f}  static err {float((ca-cb).norm()):.3f}")
