import torch, time
from gsplat import rasterization
t = time.time()
N = 1000; d = "cuda"
means = torch.randn(N, 3, device=d) + torch.tensor([0, 0, 5.0], device=d)
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=d), dim=-1)
scales = torch.full((N, 3), 0.05, device=d); opac = torch.full((N,), 0.8, device=d)
cols = torch.rand(N, 3, device=d, requires_grad=True)
K = torch.tensor([[[300., 0, 160], [0, 300., 120], [0, 0, 1]]], device=d)
img, alpha, _ = rasterization(means, quats, scales, opac, cols, torch.eye(4, device=d)[None], K, 320, 240)
img.sum().backward()
print("gsplat OK", img.shape, float(alpha.mean()), f"{time.time()-t:.1f}s")
