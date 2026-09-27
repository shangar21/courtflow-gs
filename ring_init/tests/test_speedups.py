"""Fused SSIM, fused ARAP rotations and bias-corrected SelectiveAdam match their references."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_fused_ssim_matches_torch():
    from ring_init.gs import fused_ssim
    from ring_init.gs.train import ssim_map, ssim_map_torch, dssim_loss
    if not fused_ssim.available(): pytest.skip("fused-ssim extension unavailable")
    torch.manual_seed(0)
    x = torch.rand(3, 70, 90, device="cuda", requires_grad=True); y = torch.rand(3, 70, 90, device="cuda")
    w = torch.rand(3, 70, 90, device="cuda"); m = torch.rand(70, 90, device="cuda") > 0.4
    a = ssim_map(x, y); (a * w).sum().backward(); ga = x.grad.clone(); x.grad = None
    b = ssim_map_torch(x, y); (b * w).sum().backward(); gb = x.grad.clone(); x.grad = None
    assert (a - b).abs().max() < 1e-4
    assert (ga - gb).norm() / gb.norm() < 1e-3
    mm = m.float()[None]
    ref = ((1 - ssim_map_torch(x * mm, y * mm)) * mm).mean()
    assert abs(float(dssim_loss(x, y, m)) - float(ref)) < 1e-5


@cuda
def test_fused_arap_rotation_matches_reference():
    from ring_init.deform import mls
    from ring_init.deform.mls_ref import mls_rotation as ref
    if not mls.kernel_available(): pytest.skip("MLS extension unavailable")
    torch.manual_seed(3); M = 300
    rest = torch.randn(M, 3, device="cuda", dtype=torch.float64) * 0.1
    nbr = torch.cat((torch.arange(M, device="cuda")[:, None], torch.cdist(rest, rest).topk(7, largest=False).indices[:, 1:]), 1)
    w = torch.full(nbr.shape, 1 / nbr.shape[1], device="cuda", dtype=torch.float64)
    t = torch.randn(M, 3, device="cuda", dtype=torch.float64) * 0.02; G = torch.randn(M, 3, 3, device="cuda", dtype=torch.float64)
    tr = t.clone().requires_grad_(True); Rr = ref(rest, nbr, w, tr)[0]; (Rr * G).sum().backward()
    tk = t.float().clone().requires_grad_(True); Rk = mls.mls_rotation(rest.float(), nbr, w.float(), tk); (Rk * G.float()).sum().backward()
    assert (Rk.double() - Rr).abs().max() < 1e-5
    assert (tk.grad.double() - tr.grad).norm() / tr.grad.norm() < 1e-4


@cuda
def test_bias_corrected_selective_adam_equals_adam_when_all_visible():
    from ring_init.gs.train import BiasCorrectedSelectiveAdam
    torch.manual_seed(1)
    p1 = torch.nn.Parameter(torch.randn(50, 3, device="cuda")); p2 = torch.nn.Parameter(p1.detach().clone())
    a = torch.optim.Adam([{"params": p1, "lr": 0.01}], eps=1e-15)
    b = BiasCorrectedSelectiveAdam([{"params": p2, "lr": 0.01}], eps=1e-15, betas=(0.9, 0.999))
    vis = torch.ones(50, dtype=torch.bool, device="cuda")
    for _ in range(20):
        g = torch.randn(50, 3, device="cuda")
        p1.grad = g.clone(); p2.grad = g.clone(); a.step(); b.step(vis)
    assert torch.allclose(p1, p2, atol=1e-5)
    # invisible Gaussians are left untouched
    before = p2.detach().clone(); p2.grad = torch.randn(50, 3, device="cuda"); vis[:25] = False; b.step(vis)
    assert torch.equal(p2[:25], before[:25]) and not torch.equal(p2[25:], before[25:])
