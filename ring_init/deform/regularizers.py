"""Control-point regularizers. Residuals are divided by the rest edge length (ARAP) or the mean
control spacing (temporal) so weights are unit-free and independent of the scene scale."""
from __future__ import annotations
import torch
from ring_init.deform.mls_ref import quat_to_matrix


def arap(rest: torch.Tensor, translations: torch.Tensor, neighbors: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """sum over edges ||(p'_a - p'_b) - R_a (p_a - p_b)||^2 / ||p_a - p_b||^2 (mean over edges).
    rotations: [M,3,3] (from q_a, or from each control's local MLS fit when rotations are off)."""
    p = rest; q = rest + translations
    e = p[:, None] - p[neighbors]
    delta = q[:, None] - q[neighbors]
    expected = torch.einsum("mij,mkj->mki", rotations, e)
    return ((delta - expected).square().sum(-1) / e.square().sum(-1).clamp_min(1e-12)).mean()


def cp_rotations_from_quaternions(q: torch.Tensor) -> torch.Tensor:
    return quat_to_matrix(q)


def cp_rotations_from_mls(rest: torch.Tensor, translations: torch.Tensor, neighbors: torch.Tensor,
                          neighborhood: torch.Tensor | None = None, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Rotation of each control's own neighbourhood (itself + graph neighbours, uniform weights)."""
    from ring_init.deform.mls import mls_rotation
    # The graph topology never changes within a tracker.  SceneTracker supplies the cached
    # self-inclusive neighbourhood/weights to avoid allocating them on every iteration.
    nbr = neighborhood if neighborhood is not None else torch.cat((torch.arange(len(rest), device=rest.device)[:, None], neighbors), 1)
    w = weights if weights is not None else torch.full(nbr.shape, 1.0 / nbr.shape[1], device=rest.device, dtype=rest.dtype)
    return mls_rotation(rest, nbr, w, translations)   # fused CUDA kernel when available


def temporal_acceleration(current: torch.Tensor, previous: torch.Tensor, before_previous: torch.Tensor, spacing: float) -> torch.Tensor:
    """||t(f) - 2 t(f-1) + t(f-2)||^2 / spacing^2 (mean over controls)."""
    return ((current - 2 * previous + before_previous).square().sum(-1) / spacing ** 2).mean()
