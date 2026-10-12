import numpy as np

from pointcept.datasets.transform import GridSample

from stsd.datasets.transform import RingWindowCrop, StrengthToColor


def toy_scene(n=20000, length=10.0, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, length, n)
    c = rng.uniform(0, 15.9, n)
    unroll = np.stack([s, c, rng.normal(0, 0.01, n)], 1).astype(np.float32)
    return dict(
        coord=np.stack([s, np.cos(c / 2.5) * 2.5, np.sin(c / 2.5) * 2.5], 1).astype(np.float32),
        unroll=unroll,
        strength=rng.uniform(0, 1.5, (n, 1)).astype(np.float32),
        normal=np.tile([0, 0, 1], (n, 1)).astype(np.float32),
        segment=rng.integers(0, 12, n).astype(np.int32),
        index_valid_keys=["coord", "unroll", "strength", "normal", "segment", "color"],
    )


def test_random_window_has_requested_length():
    out = RingWindowCrop(length=2.5, mode="random")(toy_scene())
    s = out["unroll"][:, 0]
    assert s.max() - s.min() <= 2.5 and len(out["coord"]) == len(out["segment"])


def test_sliding_windows_cover_every_test_fragment_point():
    data = StrengthToColor()(toy_scene())
    fragments = GridSample(grid_size=0.05, mode="test", return_grid_coord=True)(data)
    crop = RingWindowCrop(length=2.5, mode="all", stride=1.25)
    for fragment in fragments:
        parts = crop(fragment)
        covered = np.unique(np.concatenate([p["index"] for p in parts]))
        assert np.array_equal(covered, np.unique(fragment["index"]))
        for p in parts:  # every per-point array stays aligned with `index`
            assert len(p["index"]) == len(p["coord"]) == len(p["grid_coord"])


def test_strength_to_color_replicates_clipped_intensity():
    out = StrengthToColor(max_value=1.0)(toy_scene())
    assert out["color"].shape[1] == 3 and out["color"].max() <= 255.0
    assert np.allclose(out["color"][:, 0], out["color"][:, 2])


def test_legacy_intensity_reproduces_raw_over_4096():
    from stsd.datasets.transform import LegacyIntensityToColor

    data = dict(name="round-1-12_1", strength=np.array([[1.0], [0.5]], np.float32))
    out = LegacyIntensityToColor(scales=dict(circular=183.0))(data)
    assert np.allclose(out["color"][:, 0], np.array([183.0, 91.5]) / 4096 * 255)
