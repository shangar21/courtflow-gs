import numpy as np
from ring_init.io.crop import from_crop, square_box, to_crop


def test_crop_round_trip_and_bounds():
    box = square_box(np.array([100, 50]), np.array([140, 150]), 0.35, 96, 1874, 1054)
    assert box[2] - box[0] == box[3] - box[1] and box[0] >= 0 and box[3] <= 1054
    uv = np.array([[101.3, 60.7], [139.9, 149.2]])
    assert np.allclose(from_crop(to_crop(uv, box, 512), box, 512), uv)
    # Pixel centre convention: crop pixel centres map to evenly spaced source positions.
    edge = square_box(np.array([1850, 1000]), np.array([1870, 1050]), 0.35, 96, 1874, 1054)
    assert edge[2] <= 1874 and edge[3] <= 1054
