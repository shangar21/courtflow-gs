"""Fused CUDA MLS kernel vs the float64 PyTorch reference."""
import pytest
import torch
from ring_init.deform.mls_ref import rigid_mls
from ring_init.tests.mls_fixtures import conditioning, make_config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _kernel():
    from ring_init.deform.mls import kernel_available, fused_mls
    if not kernel_available(): pytest.skip("MLS CUDA extension unavailable")
    return fused_mls


def _run(kind, K, blend, use_sh, group=1, N=3000, seed=0):
    fused_mls = _kernel()
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=N, K=K, kind=kind, seed=seed, device="cuda")
    sh = sh1 if use_sh else None
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    gm, gq, gs = (torch.randn(*s, device="cuda", dtype=torch.float64, generator=g) for s in ((N, 3), (N, 4), (N, 3, 3)))
    # Well-posed Gaussians only: float32 accuracy of the polar factor is ~eps32 / cond, so below
    # cond ~ 0.05 a float32 kernel cannot meet 1e-5 (degenerate cases: finiteness test below).
    good = conditioning(rest, nbr, w, t) > 5e-2
    # Reference restricted to well-posed Gaussians (its gradient is the oracle there).
    tr = t.clone().requires_grad_(True); qr = q.clone().requires_grad_(True)
    xr, orr, sr = rigid_mls(means[good], quats[good], None if sh is None else sh[good], rest, nbr[good], w[good], tr, qr, blend, 0.5)
    ((xr * gm[good]).sum() + (orr * gq[good]).sum() + ((sr * gs[good]).sum() if use_sh else 0)).backward()
    tk = t.float().clone().requires_grad_(True); qk = q.float().clone().requires_grad_(True)
    xk, ok, sk = fused_mls(means[good], quats[good], None if sh is None else sh[good], rest, nbr[good], w[good], tk, qk, blend, 0.5, 1e-6, group)
    ((xk * gm[good].float()).sum() + (ok * gq[good].float()).sum() + ((sk * gs[good].float()).sum() if use_sh else 0)).backward()
    return (xk, ok, sk, tk.grad, qk.grad), (xr, orr, sr, tr.grad, qr.grad)


def _rel(a, b): return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


@cuda
@pytest.mark.parametrize("kind", ["random", "planar", "reflect"])
@pytest.mark.parametrize("K", [4, 8, 16])
@pytest.mark.parametrize("blend", [False, True])
def test_kernel_matches_reference(kind, K, blend):
    (xk, ok, sk, gtk, gqk), (xr, orr, sr, gtr, gqr) = _run(kind, K, blend, True)
    assert (xk.double() - xr).abs().max() < 1e-5
    assert (ok.double() - orr).abs().max() < 1e-4       # unit quaternions (float32 conversion)
    assert (sk.double() - sr).abs().max() < 1e-4        # O(1) SH coefficients
    assert _rel(gtk, gtr) < 1e-4
    if blend: assert _rel(gqk, gqr) < 1e-4


@cuda
@pytest.mark.parametrize("group", [2, 4, 8, 16])
def test_group_variants_match(group):
    (xk, ok, sk, gtk, gqk), (xr, orr, sr, gtr, gqr) = _run("random", 16, True, True, group=group)
    assert (xk.double() - xr).abs().max() < 1e-5 and _rel(gtk, gtr) < 1e-4 and _rel(gqk, gqr) < 1e-4


@cuda
def test_kernel_without_sh_and_identity():
    fused_mls = _kernel()
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=500, K=8, device="cuda", dtype=torch.float32)
    t0 = torch.zeros_like(t, requires_grad=True)
    x, o, s = fused_mls(means, quats, None, rest, nbr, w, t0)
    assert s is None and (x - means).abs().max() < 1e-5
    canon = quats * torch.where(quats[:, :1] < 0, -1.0, 1.0)
    assert (o - canon).abs().max() < 1e-5


@cuda
def test_degenerate_neighbourhoods_stay_finite():
    fused_mls = _kernel()
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=3000, K=4, kind="planar", device="cuda", dtype=torch.float32)
    t = t.clone().requires_grad_(True); q = q.clone().requires_grad_(True)
    x, o, s = fused_mls(means, quats, sh1, rest, nbr, w, t, q, True, 0.5)
    (x.sum() + o.sum() + s.sum()).backward()
    assert torch.isfinite(x).all() and torch.isfinite(o).all() and torch.isfinite(t.grad).all() and torch.isfinite(q.grad).all()
