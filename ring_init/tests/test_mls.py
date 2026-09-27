import math
import torch
import torch.nn.functional as F
from ring_init.deform.mls_ref import SH1_PERM, blend_twist, matrix_to_quat, quat_to_matrix, rigid_mls
from ring_init.tests.mls_fixtures import make_config


def test_identity_state_returns_inputs():
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=300, K=8)
    t = torch.zeros_like(t); q = torch.zeros_like(q); q[:, 0] = 1
    x, o, s = rigid_mls(means, quats, sh1, rest, nbr, w, t, q, blend=True, beta=0.5)
    canon = quats * torch.where(quats[:, :1] < 0, -1.0, 1.0)
    assert torch.allclose(x, means, atol=1e-10) and torch.allclose(o, canon, atol=1e-10) and torch.allclose(s, sh1, atol=1e-10)


def test_global_rigid_motion_is_reproduced():
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=300, K=8)
    ang = 0.7; Rg = quat_to_matrix(torch.tensor([math.cos(ang / 2), 0, math.sin(ang / 2), 0], dtype=torch.float64))
    tr = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    t = rest @ Rg.T + tr - rest
    x, o, _ = rigid_mls(means, quats, None, rest, nbr, w, t)
    assert torch.allclose(x, means @ Rg.T + tr, atol=1e-9)
    assert torch.allclose(quat_to_matrix(o), Rg @ quat_to_matrix(quats), atol=1e-9)


def test_reflection_fix_gives_proper_rotations():
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=300, K=8, kind="reflect")
    from ring_init.deform.mls_ref import mls_rotation
    R, _, _ = mls_rotation(rest, nbr, w, t)
    assert torch.allclose(torch.det(R), torch.ones(len(R), dtype=R.dtype), atol=1e-9)


def test_matrix_quat_round_trip():
    q = F.normalize(torch.randn(500, 4, dtype=torch.float64), dim=-1); q = q * torch.where(q[:, :1] < 0, -1.0, 1.0)
    assert torch.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


def test_sh_band1_rotation_matches_gsplat_convention():
    """Colour of the rotated Gaussian seen along R d equals the original seen along d."""
    from gsplat.cuda._torch_impl import _spherical_harmonics
    torch.manual_seed(0)
    n = 200
    R = quat_to_matrix(F.normalize(torch.randn(n, 4, dtype=torch.float64), dim=-1))
    c = torch.randn(n, 4, 3, dtype=torch.float64)
    c_rot = c.clone(); c_rot[:, 1:] = (SH1_PERM.double() @ R @ SH1_PERM.double().T) @ c[:, 1:]
    d = F.normalize(torch.randn(n, 3, dtype=torch.float64), dim=-1)
    a = _spherical_harmonics(1, d, c); b = _spherical_harmonics(1, torch.einsum("nij,nj->ni", R, d), c_rot)
    assert torch.allclose(a, b, atol=1e-10)


def test_reference_gradcheck_float64():
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=12, M=16, K=6)
    t = t.clone().requires_grad_(True); q = q.clone().requires_grad_(True)
    def f(t_, q_):
        x, o, s = rigid_mls(means, quats, sh1, rest, nbr, w, t_, q_, blend=True, beta=0.4)
        return x, o, s
    assert torch.autograd.gradcheck(f, (t, q), eps=1e-6, atol=1e-5)


def test_blend_twist_identity_and_sign_invariance():
    means, quats, sh1, rest, nbr, w, t, q = make_config(N=50, K=8)
    tau = blend_twist(q, nbr, w, 0.5); tau_neg = blend_twist(-q, nbr, w, 0.5)
    assert torch.allclose(tau, tau_neg, atol=1e-12)
    ident = torch.zeros_like(q); ident[:, 0] = 1
    assert torch.allclose(blend_twist(ident, nbr, w, 0.7)[:, 0], torch.ones(50, dtype=q.dtype))
