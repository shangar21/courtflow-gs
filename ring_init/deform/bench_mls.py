"""Benchmark the fused MLS kernel against the PyTorch reference.

    PYTHONPATH=. python -m ring_init.deform.bench_mls [--sizes 100000 500000] [--K 8] [--controls 7168]
"""
from __future__ import annotations
import argparse
import json
import torch
import torch.nn.functional as F


def config(N: int, M: int, K: int, device: str = "cuda", sort: bool = True, seed: int = 0):
    """Person-like layout: control points in 14 clusters, Gaussians around them, sorted by nearest CP
    (as Stage B does) so warp neighbours share control points."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    centres = torch.rand(14, 3, generator=g) * 2
    rest = (centres.repeat_interleave(M // 14 + 1, 0)[:M] + 0.05 * torch.randn(M, 3, generator=g)).to(device)
    means = (rest[torch.randint(0, M, (N,), generator=g).to(device)] + 0.01 * torch.randn(N, 3, generator=g).to(device))
    nbr = torch.cat([torch.cdist(c, rest).topk(K, largest=False).indices for c in means.split(50_000)])
    if sort:
        order = torch.argsort(nbr[:, 0]); means, nbr = means[order], nbr[order]
    dist = (means[:, None] - rest[nbr]).norm(dim=-1)
    sigma = torch.cdist(rest, rest).topk(4, largest=False).values[:, 1:].mean(1)
    w = torch.exp(-dist ** 2 / (2 * sigma[nbr] ** 2)); w = w / w.sum(1, keepdim=True)
    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    sh1 = torch.randn(N, 3, 3, device=device)
    t = 0.01 * torch.randn(M, 3, device=device); q = F.normalize(torch.tensor([1., 0, 0, 0], device=device) + 0.05 * torch.randn(M, 4, device=device), dim=-1)
    return means, quats, sh1, rest, nbr.int(), w, t, q


def timeit(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps): fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main() -> None:
    from ring_init.deform.mls import fused_mls
    from ring_init.deform.mls_ref import rigid_mls
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 500_000])
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--controls", type=int, default=7168)
    ap.add_argument("--groups", type=int, nargs="+", default=[1]); ap.add_argument("--blend", action="store_true")
    ap.add_argument("--no-reference", action="store_true"); ap.add_argument("--json")
    args = ap.parse_args(); rows = []
    for N in args.sizes:
        means, quats, sh1, rest, nbr, w, t, q = config(N, args.controls, args.K)
        tq = t.clone().requires_grad_(True); qq = q.clone().requires_grad_(True)
        gm, gq, gs = torch.randn_like(means), torch.randn_like(quats), torch.randn_like(sh1)

        def run(fn, **kw):
            def fwd():
                with torch.no_grad(): fn(means, quats, sh1, rest, nbr, w, t, q, args.blend, 0.5, **kw)
            def fwdbwd():
                x, o, s = fn(means, quats, sh1, rest, nbr, w, tq, qq, args.blend, 0.5, **kw)
                torch.autograd.backward((x, o, s), (gm, gq, gs)); tq.grad = None; qq.grad = None
            return timeit(fwd), timeit(fwdbwd)
        for G in args.groups:
            f, fb = run(fused_mls, eps=1e-6, group=G)
            rows.append({"N": N, "K": args.K, "impl": f"kernel G={G}", "fwd_ms": f, "fwd_bwd_ms": fb})
        if not args.no_reference:
            f, fb = run(rigid_mls)
            rows.append({"N": N, "K": args.K, "impl": "reference", "fwd_ms": f, "fwd_bwd_ms": fb})
        for r in rows:
            if r["N"] == N: print(f"N={N:7d} K={args.K:2d} {r['impl']:12s} fwd {r['fwd_ms']:8.3f} ms  fwd+bwd {r['fwd_bwd_ms']:8.3f} ms")
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as fh: json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
