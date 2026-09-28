from ring_init.deform.scene_track import densify_cap


def test_uncapped_growth_compounds_per_keyframe_as_before():
    assert densify_cap(n_trained=1000, growth=0.05, base=800, cap=0.0) == 1050


def test_cap_is_relative_to_the_frame0_model_not_the_current_size():
    # 70 keyframes at 5% would compound to ~30x; the cap holds the total at base * (1 + cap).
    assert densify_cap(n_trained=1000, growth=0.05, base=800, cap=0.5) == 1050   # below cap: per-keyframe limit
    assert densify_cap(n_trained=1190, growth=0.05, base=800, cap=0.5) == 1200   # clipped to 800 * 1.5
    assert densify_cap(n_trained=3000, growth=0.05, base=800, cap=0.5) == 3000   # never forces pruning


def test_absolute_cap_bounds_the_total_in_gaussians():
    assert densify_cap(n_trained=1_950_000, growth=0.05, base=800_000, cap=0.0, max_total=2_000_000) == 2_000_000
    assert densify_cap(n_trained=1_000_000, growth=0.05, base=800_000, cap=0.0, max_total=2_000_000) == 1_050_000
    assert densify_cap(n_trained=2_500_000, growth=0.05, base=800_000, cap=0.0, max_total=2_000_000) == 2_500_000  # no forced pruning
