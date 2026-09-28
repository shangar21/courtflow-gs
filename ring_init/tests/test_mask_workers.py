from ring_init.stage_b import sam2_worker_count

GB = 2 ** 30


def test_workers_limited_by_request_and_camera_count():
    assert sam2_worker_count(4, cameras=12, frames=700, available_bytes=200 * GB) == 4
    assert sam2_worker_count(4, cameras=2, frames=700, available_bytes=200 * GB) == 2
    assert sam2_worker_count(1, cameras=12, frames=700, available_bytes=200 * GB) == 1


def test_workers_limited_by_host_memory_for_the_buffered_clip():
    # 700 frames x 3 x 1024^2 float32 = 8.2 GiB per camera, plus headroom.
    assert sam2_worker_count(8, cameras=12, frames=700, available_bytes=40 * GB) == 3
    assert sam2_worker_count(8, cameras=12, frames=700, available_bytes=5 * GB) == 1
    assert sam2_worker_count(8, cameras=12, frames=30, available_bytes=40 * GB) == 8
