from ring_init.config import PIXEL_LENGTH_FIELDS, Config


def test_default_capture_scale_leaves_pixel_params_unchanged():
    cfg = Config(); before = {k: getattr(cfg, k) for k in (*PIXEL_LENGTH_FIELDS, "min_mask_area_px")}
    cfg.rescale_pixel_params()
    assert {k: getattr(cfg, k) for k in before} == before


def test_native_4k_doubles_lengths_and_quadruples_areas():
    ref = Config(); cfg = Config(capture_scale=1.0); cfg.rescale_pixel_params()
    for name in PIXEL_LENGTH_FIELDS:
        assert getattr(cfg, name) == 2 * getattr(ref, name)
        assert type(getattr(cfg, name)) is type(getattr(ref, name))
    assert cfg.min_mask_area_px == 4 * ref.min_mask_area_px
