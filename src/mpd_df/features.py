"""Leakage-safe hand-crafted EEG features."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .constants import BANDS


def bandpower_features(
    windows: np.ndarray,
    sfreq: float,
    channel_names: list[str] | tuple[str, ...],
    bands: dict[str, tuple[float, float]] = BANDS,
) -> tuple[np.ndarray, list[str]]:
    """Extract absolute/relative band powers, ratios, and spectral entropy."""

    data = np.asarray(windows)
    if data.ndim != 3:
        raise ValueError("windows must have shape (batch, channels, samples)")
    if data.shape[1] != len(channel_names):
        raise ValueError("channel_names length does not match the channel axis")
    frequencies, psd = welch(
        data,
        fs=sfreq,
        axis=-1,
        nperseg=min(data.shape[-1], int(round(sfreq))),
    )
    positive = frequencies > 0
    total = np.trapezoid(psd[..., positive], frequencies[positive], axis=-1)
    total = np.maximum(total, np.finfo(float).eps)

    matrices: list[np.ndarray] = []
    names: list[str] = []
    absolute: dict[str, np.ndarray] = {}
    for band, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)
        absolute[band] = power
        matrices.extend([power, power / total])
        for prefix in ("abs", "rel"):
            names.extend(f"{channel}_{prefix}_{band}" for channel in channel_names)

    denominator = np.maximum(absolute["beta"], np.finfo(float).eps)
    ratios = {
        "theta_beta": absolute["theta"] / denominator,
        "alpha_beta": absolute["alpha"] / denominator,
        "theta_alpha_beta": (absolute["theta"] + absolute["alpha"]) / denominator,
    }
    for ratio_name, values in ratios.items():
        matrices.append(values)
        names.extend(f"{channel}_{ratio_name}" for channel in channel_names)

    normalized_psd = psd[..., positive] / np.maximum(
        psd[..., positive].sum(axis=-1, keepdims=True),
        np.finfo(float).eps,
    )
    entropy = -(normalized_psd * np.log2(np.maximum(normalized_psd, 1e-15))).sum(axis=-1)
    entropy /= np.log2(normalized_psd.shape[-1])
    matrices.append(entropy)
    names.extend(f"{channel}_spectral_entropy" for channel in channel_names)

    return np.concatenate(matrices, axis=1).astype(np.float32), names

