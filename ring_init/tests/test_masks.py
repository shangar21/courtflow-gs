"""Synthetic ring scene: occupancy carving must recover cross-view-consistent instance IDs."""
import numpy as np
from ring_init.geom.floor import Floor
from ring_init.io.calib import Camera
from ring_init.masks.instances import find_instances, splat_silhouette


def ring_cameras(n=12, radius=6.0, height=3.0, size=(320, 240)):
    cams = []
    for i in range(n):
        a = 2*np.pi*i/n; c = np.array([radius*np.cos(a), height, radius*np.sin(a)])
        z = -c / np.linalg.norm(c); x = np.cross([0, 1., 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack((x, y, z)); K = np.array([[300., 0, size[0]/2], [0, 300., size[1]/2], [0, 0, 1]])
        cams.append(Camera(f"c{i}", K, np.zeros(0), R, -R @ c, *size))
    return cams


def cylinder(center, radius=0.25, height=1.8, n=4000, seed=0):
    rng = np.random.default_rng(seed); r = radius*np.sqrt(rng.random(n)); t = rng.random(n)*2*np.pi
    return np.stack((center[0] + r*np.cos(t), rng.random(n)*height, center[1] + r*np.sin(t)), 1)


def test_occupancy_recovers_consistent_ids():
    cams = ring_cameras(); people = [cylinder((-1.2, 0.0), seed=1), cylinder((1.0, 0.8), seed=2), cylinder((0.3, -1.3), seed=3)]
    unions = []
    for cam in cams:
        m = np.zeros((cam.height, cam.width), bool)
        for p in people: m |= splat_silhouette(p, cam, 1, device="cpu")[0]
        unions.append(m)
    floor = Floor(np.array([0, -1., 0])*-1, 0.0, np.zeros(3), np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([-2.5, -2.5]), np.array([2.5, 2.5]))
    instances, _ = find_instances(cams, unions, floor, voxel=0.08, height=2.2, margin=0.2, min_views=4, allowed_misses=1, min_height=1.0, min_voxels=10, split_footprint=5.0, device="cpu")
    assert len(instances) == 3
    # Each recovered instance contains exactly one true person centre (identity is 3D, hence
    # consistent in every view by construction).
    for inst in instances:
        d = [np.linalg.norm(inst.center[[0, 2]] - p.mean(0)[[0, 2]]) for p in people]
        assert min(d) < 0.15
    assert len({int(np.argmin([np.linalg.norm(i.center[[0, 2]] - p.mean(0)[[0, 2]]) for p in people])) for i in instances}) == 3
