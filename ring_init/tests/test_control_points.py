import torch
from ring_init.deform.control_points import build_topology
from ring_init.deform.track import FrameState, warm_start


def test_topology_is_per_instance_and_normalized():
    torch.manual_seed(0)
    a = torch.randn(400, 3) * 0.1; b = torch.randn(300, 3) * 0.1 + torch.tensor([1.0, 0, 0])
    topo = build_topology([(3, a), (7, b)], controls=32, gaussian_knn=8, graph_knn=6, sigma_knn=3)
    assert topo.rest.shape == (64, 3)
    assert torch.allclose(topo.gaussian_weights.sum(1), torch.ones(700), atol=1e-5)
    owner = topo.cp_instance
    assert (owner[topo.gaussian_neighbors[:400]] == 3).all() and (owner[topo.gaussian_neighbors[400:]] == 7).all()
    assert (owner[topo.graph_neighbors] == owner[:, None]).all()   # graph never crosses instances
    # each Gaussian's neighbours are its nearest controls of its own instance
    d = torch.cdist(a, topo.rest[:32]); assert torch.equal(topo.gaussian_neighbors[:400].sort(1).values, d.topk(8, largest=False).indices.sort(1).values)


def test_constant_velocity_warm_start():
    q = torch.tensor([[1.0, 0, 0, 0]])
    s1 = FrameState(torch.tensor([[0.0, 0, 0]]), q); s2 = FrameState(torch.tensor([[1.0, 2, 3]]), q)
    w = warm_start(s2, s1, True)
    assert torch.allclose(w.t, torch.tensor([[2.0, 4, 6]])) and torch.allclose(w.q, q)
    assert torch.allclose(warm_start(s2, s1, False).t, s2.t)
