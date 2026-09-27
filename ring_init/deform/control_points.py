"""Per-instance control points (never shared across instances), Gaussian->CP RBF weights and the
CP kNN graph, concatenated into one scene-level topology with global CP indices."""
from __future__ import annotations
from dataclasses import dataclass
import torch


def farthest_point_sample(x: torch.Tensor, count: int, seed_index: int = 0) -> torch.Tensor:
    count = min(count, len(x))
    idx = torch.empty(count, dtype=torch.long, device=x.device); idx[0] = seed_index
    distance = torch.full((len(x),), float("inf"), device=x.device)
    for i in range(1, count):
        torch.minimum(distance, ((x - x[idx[i-1]]) ** 2).sum(-1), out=distance); idx[i] = distance.argmax()
    return idx


@dataclass
class ControlTopology:
    rest: torch.Tensor               # [M,3] control rest positions
    gaussian_neighbors: torch.Tensor  # [N,K] global CP indices
    gaussian_weights: torch.Tensor   # [N,K] rows sum to 1
    graph_neighbors: torch.Tensor    # [M,G] global CP indices (same instance)
    cp_instance: torch.Tensor        # [M] instance id of each CP
    sigma: torch.Tensor              # [M]

    @property
    def spacing(self) -> float:
        """Mean distance between neighbouring controls (normalizes ARAP / temporal terms)."""
        return float((self.rest[:, None] - self.rest[self.graph_neighbors]).norm(dim=-1).mean())


def build_instance(means: torch.Tensor, controls: int, gaussian_knn: int, graph_knn: int, sigma_knn: int):
    idx = farthest_point_sample(means, controls); rest = means[idx].detach().clone(); m = len(rest)
    k = min(gaussian_knn, m)
    d = torch.cdist(means, rest); nearest = d.topk(k, largest=False).indices
    cd = torch.cdist(rest, rest)
    sigma = cd.topk(min(sigma_knn + 1, m), largest=False).values[:, 1:].mean(1).clamp_min(1e-9)
    local = d.gather(1, nearest)
    logw = -(local ** 2) / (2 * sigma[nearest] ** 2)
    w = torch.softmax(logw, 1)       # = exp(.)/sum exp(.), without underflow for far Gaussians
    graph = cd.topk(min(graph_knn + 1, m), largest=False).indices[:, 1:]
    return rest, nearest, w, graph, sigma


def build_topology(instances: list[tuple[int, torch.Tensor]], controls: int, gaussian_knn: int, graph_knn: int, sigma_knn: int) -> ControlTopology:
    """instances: [(instance_id, canonical means [N_i,3])] in the order the Gaussians are concatenated."""
    rests, nbrs, ws, graphs, owners, sigmas = [], [], [], [], [], []; offset = 0
    for ident, means in instances:
        rest, nearest, w, graph, sigma = build_instance(means, controls, gaussian_knn, graph_knn, sigma_knn)
        rests.append(rest); nbrs.append(nearest + offset); ws.append(w); graphs.append(graph + offset)
        owners.append(torch.full((len(rest),), ident, dtype=torch.long, device=rest.device)); sigmas.append(sigma); offset += len(rest)
    return ControlTopology(torch.cat(rests), torch.cat(nbrs), torch.cat(ws), torch.cat(graphs), torch.cat(owners), torch.cat(sigmas))


# --------------------------------------------------------------------------- scene-wide (Stage B v2)
@dataclass
class SceneControls:
    """Control points for every group (background = 0, persons = instance id). Positions are explicit
    state; Gaussian->control neighbours are always taken within the Gaussian's own group."""
    pos: torch.Tensor          # [M,3] current control positions
    group: torch.Tensor        # [M] group id
    graph: torch.Tensor        # [M,G] neighbours within the group
    sigma: torch.Tensor        # [M]
    nbr: torch.Tensor          # [N,K]
    w: torch.Tensor            # [N,K]

    @property
    def spacing(self) -> float:
        return float((self.pos[:, None] - self.pos[self.graph]).norm(dim=-1).mean())


def _knn_chunked(x: torch.Tensor, ref: torch.Tensor, k: int, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    ds, idx = [], []
    for s in range(0, len(x), chunk):
        d = torch.cdist(x[s:s+chunk], ref); v, i = d.topk(k, largest=False); ds.append(v); idx.append(i)
    return torch.cat(ds), torch.cat(idx)


def bind_gaussians(means: torch.Tensor, groups: torch.Tensor, ctrl_pos: torch.Tensor, ctrl_group: torch.Tensor, sigma: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K nearest same-group controls and normalized RBF weights exp(-d^2 / 2 sigma_j^2) per Gaussian."""
    nbr = torch.zeros(len(means), k, dtype=torch.long, device=means.device); w = torch.zeros(len(means), k, device=means.device)
    for g in torch.unique(groups).tolist():
        gi = torch.nonzero(groups == g)[:, 0]; ci = torch.nonzero(ctrl_group == g)[:, 0]
        if len(ci) == 0: raise RuntimeError(f"group {g} has Gaussians but no control points")
        kk = min(k, len(ci)); d, j = _knn_chunked(means[gi], ctrl_pos[ci], kk)
        glob = ci[j]; logw = -(d ** 2) / (2 * sigma[glob] ** 2)
        if kk < k:  # pad with the nearest control and zero weight
            glob = torch.cat((glob, glob[:, :1].expand(-1, k - kk)), 1); logw = torch.cat((logw, torch.full((len(gi), k - kk), -1e9, device=means.device)), 1)
        nbr[gi] = glob; w[gi] = torch.softmax(logw, 1)
    return nbr, w


def build_scene_controls(means: torch.Tensor, groups: torch.Tensor, counts: dict[int, int], k: int, graph_knn: int, sigma_knn: int) -> SceneControls:
    pos, grp, graph, sigma = [], [], [], []; offset = 0
    for g in torch.unique(groups).tolist():
        pts = means[groups == g]; n = counts.get(g, counts.get(-1, 512))
        idx = farthest_point_sample(pts, n); p = pts[idx].detach().clone(); m = len(p)
        cd = torch.cdist(p, p)
        s = cd.topk(min(sigma_knn + 1, m), largest=False).values[:, 1:].mean(1).clamp_min(1e-9)
        gr = cd.topk(min(graph_knn + 1, m), largest=False).indices[:, 1:]
        pos.append(p); grp.append(torch.full((m,), g, dtype=torch.long, device=p.device)); graph.append(gr + offset); sigma.append(s); offset += m
    pos, grp, graph, sigma = torch.cat(pos), torch.cat(grp), torch.cat(graph), torch.cat(sigma)
    nbr, w = bind_gaussians(means, groups, pos, grp, sigma, k)
    return SceneControls(pos, grp, graph, sigma, nbr, w)
