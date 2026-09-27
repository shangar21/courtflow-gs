"""Shared random MLS configurations for the reference and kernel tests."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from ring_init.deform.mls_ref import quat_to_matrix


def make_config(N=2000, M=64, K=8, kind="random", seed=0, dtype=torch.float64, device="cpu"):
    """kind: random | planar (control points ~on a plane) | reflect (mirror motion -> det fix)."""
    g = torch.Generator().manual_seed(seed)
    rest = torch.rand(M, 3, generator=g, dtype=torch.float64)
    means = torch.rand(N, 3, generator=g, dtype=torch.float64)
    if kind == "planar":
        rest[:, 2] = 0.5 + 1e-3 * torch.randn(M, generator=g, dtype=torch.float64)
        means[:, 2] = 0.5 + 0.01 * torch.randn(N, generator=g, dtype=torch.float64)
    d = torch.cdist(means, rest); nbr = d.topk(K, largest=False).indices
    sigma = torch.cdist(rest, rest).topk(4, largest=False).values[:, 1:].mean(1)
    w = torch.exp(-d.gather(1, nbr) ** 2 / (2 * sigma[nbr] ** 2)); w = w / w.sum(1, keepdim=True)
    axis = F.normalize(torch.randn(3, generator=g, dtype=torch.float64), dim=0); ang = 0.6
    Rg = quat_to_matrix(torch.cat([torch.tensor([math.cos(ang / 2)], dtype=torch.float64), axis * math.sin(ang / 2)]))
    moved = rest @ Rg.T + 0.1 * torch.randn(3, generator=g, dtype=torch.float64) + 0.02 * torch.randn(M, 3, generator=g, dtype=torch.float64)
    if kind == "reflect":
        moved = rest * torch.tensor([1., 1., -1.], dtype=torch.float64) + 0.01 * torch.randn(M, 3, generator=g, dtype=torch.float64)
    t = moved - rest
    q = F.normalize(torch.randn(M, 4, generator=g, dtype=torch.float64) * torch.tensor([4., 1, 1, 1], dtype=torch.float64), dim=-1)
    q = q * torch.where(torch.rand(M, 1, generator=g) < 0.3, -1.0, 1.0).double()
    quats = F.normalize(torch.randn(N, 4, generator=g, dtype=torch.float64), dim=-1)
    sh1 = torch.randn(N, 3, 3, generator=g, dtype=torch.float64)
    return [x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype) for x in (means, quats, sh1, rest, nbr, w, t, q)]


def conditioning(rest, nbr, w, t) -> torch.Tensor:
    """(s2 + signed s3) / s1 of each Gaussian's MLS covariance: ~0 means the closest rotation is
    ill-posed (near-collinear neighbourhood), where any implementation's answer is arbitrary."""
    p = rest[nbr]; pd = p + t[nbr]
    ps = (w[..., None] * p).sum(1); qs = (w[..., None] * pd).sum(1)
    M = torch.einsum("nk,nki,nkj->nij", w, p - ps[:, None], pd - qs[:, None])
    U, S, Vh = torch.linalg.svd(M)
    det = torch.det(Vh.transpose(1, 2) @ U.transpose(1, 2))
    return (S[:, 1] + S[:, 2] * det) / S[:, 0]
