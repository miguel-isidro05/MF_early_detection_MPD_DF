import numpy as np

from mpd_df.preprocessing import (
    ANNOTATION_CLASSIFICATION,
    ANNOTATION_VISUALIZATION,
    MNE_EEGLAB_LIKE,
    PHYSIOLOGICAL_GLOBAL_ZSCORE,
    PHYSIOLOGICAL_MICROVOLT,
    PHYSIOLOGICAL_VALIDATION,
    PSD_CLASSIFICATION,
    preprocess_batch,
)


def test_physiological_pipeline_downsamples_and_normalizes() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(2, 4, 500))
    processed, sfreq = preprocess_batch(windows, 500.0, PHYSIOLOGICAL_VALIDATION)
    assert processed.shape == (2, 4, 200)
    assert sfreq == 200.0
    assert np.allclose(processed.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(processed.std(axis=-1), 1.0, atol=1e-4)


def test_psd_classification_preserves_physical_amplitude() -> None:
    rng = np.random.default_rng(42)
    windows = 4.0 * rng.normal(size=(2, 4, 500))
    processed, sfreq = preprocess_batch(windows, 500.0, PSD_CLASSIFICATION)
    assert processed.shape == (2, 4, 200)
    assert sfreq == 200.0
    assert not np.allclose(processed.std(axis=-1), 1.0, atol=1e-2)


def test_global_zscore_preserves_channel_relative_scale() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(2, 4, 500))
    windows[:, 0] *= 5.0
    processed, sfreq = preprocess_batch(
        windows,
        500.0,
        PHYSIOLOGICAL_GLOBAL_ZSCORE,
    )
    assert sfreq == 200.0
    assert np.allclose(processed.mean(axis=(-2, -1)), 0.0, atol=1e-5)
    assert np.allclose(processed.std(axis=(-2, -1)), 1.0, atol=1e-4)
    assert np.all(processed[:, 0].std(axis=-1) > processed[:, 1].std(axis=-1))


def test_microvolt_profile_has_model_friendly_scale() -> None:
    rng = np.random.default_rng(42)
    windows = 20e-6 * rng.normal(size=(2, 4, 500))
    processed, sfreq = preprocess_batch(
        windows,
        500.0,
        PHYSIOLOGICAL_MICROVOLT,
    )
    assert sfreq == 200.0
    assert 1.0 < float(processed.std()) < 100.0


def test_annotation_classification_downsamples_without_window_zscore() -> None:
    rng = np.random.default_rng(42)
    windows = 4.0 * rng.normal(size=(2, 4, 500))
    processed, sfreq = preprocess_batch(
        windows,
        500.0,
        ANNOTATION_CLASSIFICATION,
    )
    assert processed.shape == (2, 4, 200)
    assert sfreq == 200.0
    assert not np.allclose(processed.std(axis=-1), 1.0, atol=1e-2)


def test_annotation_profile_uses_published_49_to_51_hz_bandstop() -> None:
    sfreq = 500.0
    time = np.arange(int(20 * sfreq)) / sfreq
    low_frequency = np.sin(2 * np.pi * 10 * time)
    line_frequency = np.sin(2 * np.pi * 50 * time)
    windows = (low_frequency + line_frequency)[None, None, :]

    processed, output_sfreq = preprocess_batch(windows, sfreq, ANNOTATION_VISUALIZATION)
    retained_10_hz = np.abs(np.mean(processed[0, 0] * low_frequency)) * 2
    retained_50_hz = np.abs(np.mean(processed[0, 0] * line_frequency)) * 2

    assert output_sfreq == sfreq
    assert retained_10_hz > 0.8
    assert retained_50_hz < 0.1


def test_mne_eeglab_like_profile_uses_native_rate_and_fir_notch() -> None:
    sfreq = 500.0
    time = np.arange(int(30 * sfreq)) / sfreq
    low_frequency = np.sin(2 * np.pi * 10 * time)
    line_frequency = np.sin(2 * np.pi * 50 * time)
    windows = (low_frequency + line_frequency)[None, None, :]

    processed, output_sfreq = preprocess_batch(windows, sfreq, MNE_EEGLAB_LIKE)
    retained_10_hz = np.abs(np.mean(processed[0, 0] * low_frequency)) * 2
    retained_50_hz = np.abs(np.mean(processed[0, 0] * line_frequency)) * 2

    assert output_sfreq == sfreq
    assert retained_10_hz > 0.8
    assert retained_50_hz < 0.1
