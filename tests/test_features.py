import numpy as np

from mpd_df.features import bandpower_features


def test_bandpower_features_have_stable_shape_and_finite_values() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(4, 3, 200)).astype(np.float32)
    features, names = bandpower_features(windows, 200.0, ["F3", "C3", "O1"])
    assert features.shape == (4, 36)
    assert len(names) == 36
    assert np.isfinite(features).all()

