"""Explicit EEG preprocessing variants used by reproduction and ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos


@dataclass(frozen=True)
class PreprocessingConfig:
    name: str
    l_freq: float | None
    h_freq: float | None
    notch_freq: float | None
    target_sfreq: float | None
    notch_band: tuple[float, float] | None = None
    demean: bool = True
    normalization: str = "none"
    filter_backend: str = "scipy_iir"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REFERENCE_UNSPECIFIED = PreprocessingConfig(
    name="reference_unspecified",
    l_freq=None,
    h_freq=None,
    notch_freq=None,
    target_sfreq=None,
    demean=False,
    normalization="none",
)

PHYSIOLOGICAL_VALIDATION = PreprocessingConfig(
    name="physiological_validation",
    l_freq=1.0,
    h_freq=100.0,
    notch_freq=50.0,
    target_sfreq=200.0,
    demean=True,
    normalization="per_window_channel_zscore",
)

ANNOTATION_VISUALIZATION = PreprocessingConfig(
    name="annotation_visualization",
    l_freq=0.3,
    h_freq=35.0,
    notch_freq=None,
    target_sfreq=None,
    notch_band=(49.0, 51.0),
    demean=True,
    normalization="none",
)

# Training counterpart of the Figure 6 EEGLAB filtering. MNE's zero-double FIR
# path approximates pop_eegfiltnew's zero-phase FIR behavior without claiming
# samplewise equivalence to EEGLAB.
MNE_EEGLAB_LIKE = PreprocessingConfig(
    name="mne_eeglab_like",
    l_freq=0.3,
    h_freq=35.0,
    notch_freq=50.0,
    target_sfreq=None,
    demean=True,
    normalization="none",
    filter_backend="mne_fir_zero_double",
)


def preprocess_batch(
    windows: np.ndarray,
    sfreq: float,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, float]:
    """Preprocess (batch, channels, samples) without cross-window statistics."""

    data = np.asarray(windows, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("windows must have shape (batch, channels, samples)")
    if config.demean:
        data = data - data.mean(axis=-1, keepdims=True)
    if config.filter_backend == "mne_fir_zero_double":
        import mne

        if config.notch_band is not None:
            low, high = config.notch_band
            notch_freq = (low + high) / 2.0
            notch_width = high - low
        else:
            notch_freq = config.notch_freq
            notch_width = 2.0
        if notch_freq is not None:
            data = mne.filter.notch_filter(
                data,
                Fs=sfreq,
                freqs=np.asarray([notch_freq]),
                notch_widths=np.asarray([notch_width]),
                method="fir",
                phase="zero-double",
                verbose="ERROR",
            )
        if config.l_freq is not None or config.h_freq is not None:
            data = mne.filter.filter_data(
                data,
                sfreq=sfreq,
                l_freq=config.l_freq,
                h_freq=config.h_freq,
                method="fir",
                phase="zero-double",
                verbose="ERROR",
            )
    elif config.notch_band is not None:
        low, high = config.notch_band
        nyquist = sfreq / 2.0
        if not 0.0 < low < high < nyquist:
            raise ValueError("notch_band must lie strictly inside the Nyquist range")
        data = sosfiltfilt(
            butter(4, [low / nyquist, high / nyquist], btype="bandstop", output="sos"),
            data,
            axis=-1,
        )
    elif config.notch_freq is not None:
        b, a = iirnotch(config.notch_freq, Q=30.0, fs=sfreq)
        data = sosfiltfilt(tf2sos(b, a), data, axis=-1)
    if config.filter_backend == "scipy_iir" and (
        config.l_freq is not None or config.h_freq is not None
    ):
        nyquist = sfreq / 2.0
        if config.l_freq is None:
            wn = config.h_freq / nyquist
            btype = "lowpass"
        elif config.h_freq is None:
            wn = config.l_freq / nyquist
            btype = "highpass"
        else:
            wn = [config.l_freq / nyquist, config.h_freq / nyquist]
            btype = "bandpass"
        data = sosfiltfilt(butter(4, wn, btype=btype, output="sos"), data, axis=-1)
    output_sfreq = float(sfreq)
    if config.target_sfreq is not None and config.target_sfreq != sfreq:
        source = int(round(sfreq))
        target = int(round(config.target_sfreq))
        divisor = gcd(source, target)
        data = resample_poly(data, target // divisor, source // divisor, axis=-1)
        output_sfreq = float(config.target_sfreq)
    if config.filter_backend not in {"scipy_iir", "mne_fir_zero_double"}:
        raise ValueError(f"Unsupported filter backend: {config.filter_backend}")
    if config.normalization == "per_window_channel_zscore":
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        data = (data - mean) / np.maximum(std, np.finfo(data.dtype).eps)
    elif config.normalization != "none":
        raise ValueError(f"Unsupported normalization: {config.normalization}")
    return data.astype(np.float32), output_sfreq
