import dataclasses

import numpy as np
import pytest
import torch

from ring_init.config import Config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def _cfg(**kw):
    return dataclasses.replace(Config(), **kw)


def _viewmat(center, target=(0.0, 0.0, 0.0)):
    c = np.asarray(center, float)
    f = np.asarray(target, float) - c
    f /= np.linalg.norm(f)
    up = np.array([0, 1.0, 0])
    x = np.cross(up, f); x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.stack([x, y, f])
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = -R @ c
    return torch.tensor(m, device="cuda")


def _random_model(n=300, seed=0):
    from ring_init.gs.init_gaussians import initialize_gaussians
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)) * 0.3
    cols = rng.uniform(size=(n, 3))
    normals = rng.normal(size=(n, 3))
    model = initialize_gaussians(pts, cols, normals, 7, _cfg())
    with torch.no_grad():
        g = torch.Generator(device="cuda").manual_seed(seed)
        model.params["shN"].normal_(0, 0.1, generator=g)
        model.params["opacities"].fill_(1.0)
    return model


@cuda
def test_ssim_identity():
    from ring_init.gs.train import ssim
    x = torch.rand(3, 40, 50, device="cuda")
    assert abs(float(ssim(x, x)) - 1.0) < 1e-5
    assert float(ssim(x, torch.rand_like(x))) < 0.5


@cuda
def test_ply_round_trip(tmp_path):
    from ring_init.gs.export import load_ply, save_ply
    model = _random_model()
    save_ply(model, tmp_path / "g.ply")
    back = load_ply(tmp_path / "g.ply")
    for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        assert torch.allclose(model.params[k], back.params[k], atol=1e-6), k
    assert (back.params["instance_ids"] == 7).all()


@cuda
def test_crop_view_matches_full_render():
    from ring_init.gs.train import View, crop_view, render
    model = _random_model()
    W, H = 160, 120
    K = torch.tensor([[150.0, 0, 80], [0, 150, 60], [0, 0, 1]], device="cuda")
    vm = _viewmat([0, 0.3, -3])
    z = torch.zeros(H, W, device="cuda")
    view = View(torch.zeros(3, H, W, device="cuda"), z > 0, z, z > 0, vm, K, W, H)
    full, _, alpha_full, _ = render(model, vm, K, W, H)
    box = (37, 21, 121, 95)
    c = crop_view(view, box)
    crop, _, alpha_crop, _ = render(model, c.viewmat, c.K, c.width, c.height)
    x0, y0 = box[0] // 16 * 16, box[1] // 16 * 16
    x1, y1 = min(-(-box[2] // 16) * 16, W), min(-(-box[3] // 16) * 16, H)
    ref = full[y0:y1, x0:x1]
    assert crop.shape == ref.shape
    assert (crop - ref).abs().max() < 2e-3
    assert (alpha_crop - alpha_full[y0:y1, x0:x1]).abs().max() < 2e-3


@cuda
def test_training_smoke_loss_decreases():
    from ring_init.gs.train import View, render, train_gaussians, psnr
    torch.manual_seed(0)
    gt = _random_model(200, seed=1)
    W, H = 96, 72
    K = torch.tensor([[90.0, 0, 48], [0, 90, 36], [0, 0, 1]], device="cuda")
    views = []
    for center in ([0, 0.2, -3], [2.5, 0.2, -1.5]):
        vm = _viewmat(center)
        with torch.no_grad():
            rgb, _, alpha, _ = render(gt, vm, K, W, H)
        target = (alpha > 0.5)
        views.append(View(rgb.permute(2, 0, 1).clamp(0, 1), target, target.float(),
                          torch.zeros_like(target), vm, K, W, H))
    model = _random_model(200, seed=2)
    before = np.mean([psnr(render(model, v.viewmat, v.K, W, H)[0].permute(2, 0, 1), v.image) for v in views])
    cfg = _cfg(t0_iterations=50, t0_views_per_step=2, densify_start_iter=10, densify_every=10,
               mask_prune_every=0, drop_gaussian=True)
    stats = train_gaussians(model, views, cfg, kind="instance")
    after = np.mean([psnr(render(model, v.viewmat, v.K, W, H)[0].permute(2, 0, 1), v.image) for v in views])
    assert after > before
    assert stats["final_gaussians"] == len(model)
    assert len(model.params["instance_ids"]) == len(model.params["means"])


@cuda
def test_batched_step_matches_sequential():
    """One batched C-camera rasterization gives the same loss, parameter gradients and
    densification statistics as rendering each view separately."""
    from ring_init.gs.train import View, _step_loss, _update_grad_state, render
    gt = _random_model(200, seed=1)
    W, H = 64, 48
    K = torch.tensor([[60.0, 0, 32], [0, 60, 24], [0, 0, 1]], device="cuda")
    views = []
    for i, center in enumerate(([0, 0.2, -3], [2.5, 0.2, -1.5], [-2.5, 0.4, -1.5])):
        vm = _viewmat(center)
        with torch.no_grad():
            rgb, depth, alpha, _ = render(gt, vm, K, W, H)
        t = alpha > 0.5
        ignore = torch.zeros_like(t)
        ignore[:8] = True
        views.append(View(rgb.permute(2, 0, 1).clamp(0, 1), t, t.float(), ignore, vm, K, W, H,
                          sparse_depth=depth * (torch.rand_like(depth) > 0.7) if i != 1 else None))
    cfg = _cfg()

    # Batched: one call. Sequential: three single-view calls summed with the same weights.
    batched_model = _random_model(200, seed=2)
    loss_b, _, infos_b = _step_loss(batched_model, views, cfg, "instance", None, 0.05)
    loss_b.backward()
    state_b = {"grad2d": None, "count": None}
    for info in infos_b:
        _update_grad_state(state_b, info, len(views))

    seq_model = _random_model(200, seed=2)
    loss_s, infos_s = 0.0, []
    for v in views:
        loss, _, infos = _step_loss(seq_model, [v], cfg, "instance", None, 0.05)
        loss_s = loss_s + loss / len(views)
        infos_s += infos
    loss_s.backward()
    state_s = {"grad2d": None, "count": None}
    for info in infos_s:
        _update_grad_state(state_s, info, len(views))

    assert torch.allclose(loss_b, loss_s, rtol=1e-5, atol=1e-7)
    for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        gb, gs = batched_model.params[k].grad, seq_model.params[k].grad
        assert torch.allclose(gb, gs, rtol=1e-3, atol=1e-6), k
    assert torch.equal(state_b["count"], state_s["count"])
    assert torch.allclose(state_b["grad2d"], state_s["grad2d"], rtol=1e-3, atol=1e-6)



def _synthetic_views(gt, n_views=3, W=64, H=48):
    from ring_init.gs.train import View, render
    K = torch.tensor([[60.0, 0, W / 2], [0, 60, H / 2], [0, 0, 1]], device="cuda")
    views = []
    for a in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
        vm = _viewmat([3 * np.sin(a), 0.3, -3 * np.cos(a)])
        with torch.no_grad():
            rgb, _, alpha, _ = render(gt, vm, K, W, H)
        t = alpha > 0.5
        views.append(View(rgb.permute(2, 0, 1).clamp(0, 1), torch.ones_like(t), t.float(),
                          torch.zeros_like(t), vm, K, W, H))
    return views


@cuda
def test_frozen_composite_matches_concat_render():
    from ring_init.gs.export import concat_models
    from ring_init.gs.train import render
    trained, frozen = _random_model(150, seed=3), _random_model(120, seed=4)
    vm = _viewmat([0, 0.3, -3])
    K = torch.tensor([[60.0, 0, 32], [0, 60, 24], [0, 0, 1]], device="cuda")
    rgb, depth, alpha, info = render(trained, vm, K, 64, 48, frozen=frozen)
    ref, ref_d, ref_a, _ = render(concat_models([trained, frozen]), vm, K, 64, 48)
    assert info["n_trained"] == 150 and info["means2d"].shape[1] == 270
    assert (rgb - ref).abs().max() < 1e-5 and (alpha - ref_a).abs().max() < 1e-5


@cuda
def test_frozen_training_touches_only_trained_slice():
    from ring_init.gs.train import _step_loss, _update_grad_state, train_gaussians
    views = _synthetic_views(_random_model(200, seed=1))
    frozen = _random_model(120, seed=4)
    before = {k: frozen.params[k].detach().clone() for k in ("means", "scales", "opacities", "sh0")}
    # One step of bookkeeping: counts must equal the trained slice's per-camera visibility.
    model = _random_model(150, seed=2)
    loss, _, infos = _step_loss(model, views, _cfg(), "background", None, 0.0, frozen)
    loss.backward()
    state = {"grad2d": None, "count": None}
    for info in infos:
        _update_grad_state(state, info, len(views))
    radii = infos[0]["radii"]
    expected = (radii[:, :150] > 0).all(-1).sum(0).float() if radii.dim() == 3 else (radii[:, :150] > 0).sum(0).float()
    assert state["count"].shape == (150,) and torch.equal(state["count"], expected)
    assert all(frozen.params[k].grad is None for k in ("means", "scales", "quats", "opacities", "sh0", "shN"))
    # Full training with densification + DropGaussian: frozen model untouched, ids aligned.
    model = _random_model(150, seed=2)
    cfg = _cfg(bg_iterations=60, bg_views_per_step=3, densify_start_iter=5, densify_every=10,
               densify_grad_threshold=1e-6, mask_prune_every=20, drop_gaussian=True, drop_gaussian_rate=0.5)
    stats = train_gaussians(model, views, cfg, kind="background", frozen=frozen)
    assert sum(h["dupli"] + h["split"] for h in stats["densify_history"]) > 0
    assert len(frozen) == 120
    for k, v in before.items():
        assert torch.equal(frozen.params[k], v), k
        assert frozen.params[k].grad is None
    assert len(model.params["instance_ids"]) == len(model) == stats["final_gaussians"]


@cuda
def test_extra_prune_removes_selected():
    from ring_init.gs.train import train_gaussians
    views = _synthetic_views(_random_model(200, seed=1))
    model = _random_model(200, seed=2)
    calls = []

    def outside(means):
        calls.append(len(means))
        return means[:, 0] > 0.2

    cfg = _cfg(t0_iterations=100, densify_start_iter=5, densify_every=10, mask_prune_every=20)
    train_gaussians(model, views, cfg, kind="instance", extra_prune=outside)
    assert len(calls) >= 2                       # at least one mask-prune event + the final prune
    assert (model.params["means"][:, 0] <= 0.2).all()
