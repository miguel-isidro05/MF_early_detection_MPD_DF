import numpy as np

from mpd_df.preprocessing import PHYSIOLOGICAL_VALIDATION, preprocess_batch


def test_physiological_pipeline_downsamples_and_normalizes() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(2, 4, 500))
    processed, sfreq = preprocess_batch(windows, 500.0, PHYSIOLOGICAL_VALIDATION)
    assert processed.shape == (2, 4, 200)
    assert sfreq == 200.0
    assert np.allclose(processed.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(processed.std(axis=-1), 1.0, atol=1e-4)

