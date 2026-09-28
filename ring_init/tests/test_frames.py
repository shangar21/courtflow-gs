from ring_init.io.frames import colmap_camera_id

IMAGES_TXT = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
1 1 0 0 0 0 0 0 7 view_000.jpg

2 1 0 0 0 0 0 0 9 view_001.jpg

"""


def test_camera_id_matches_raw_capture_jpg_names(tmp_path):
    path = tmp_path / "images.txt"; path.write_text(IMAGES_TXT)
    assert colmap_camera_id(path, 1) == 9


def test_camera_id_matches_converted_png_names(tmp_path):
    path = tmp_path / "images.txt"; path.write_text(IMAGES_TXT.replace(".jpg", ".png"))
    assert colmap_camera_id(path, 0) == 7
