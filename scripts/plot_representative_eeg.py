#!/usr/bin/env python3
"""Functional recreations of MPD-DF Figures 6 and 7."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from mpd_df.constants import BANDS, EDF_CHANNELS, LABEL_NAMES
from mpd_df.dataset import (
    EDF_ASSUMED_SCALE_TO_VOLTS,
    build_alignment,
    discover_subjects,
)
from mpd_df.preprocessing import ANNOTATION_VISUALIZATION, preprocess_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subject", default="10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--channel", default="O1")
    return parser.parse_args()


def first_full_segment(labels: np.ndarray, label: int, duration: int = 30) -> int:
    mask = labels == label
    run = np.convolve(mask.astype(int), np.ones(duration, dtype=int), mode="valid")
    matches = np.flatnonzero(run == duration)
    if not matches.size:
        raise ValueError(f"No {duration}s segment for label {label}")
    return int(matches[0])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    files = next(
        files
        for files in discover_subjects(args.raw_root, include_psg=True)
        if files.subject == args.subject
    )
    alignment = build_alignment(files)
    raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
    raw.pick(list(EDF_CHANNELS))
    sfreq = float(raw.info["sfreq"])
    selections = []
    processed_by_label = {}
    context_sec = 10
    for label in range(5):
        aligned_start = first_full_segment(alignment.labels, label)
        eeg_start = alignment.eeg_offset_sec + aligned_start
        read_start = max(0, eeg_start - context_sec)
        read_end = min(raw.n_times / sfreq, eeg_start + 30 + context_sec)
        values = (
            raw.get_data(
                start=int(round(read_start * sfreq)),
                stop=int(round(read_end * sfreq)),
            )
            * EDF_ASSUMED_SCALE_TO_VOLTS
        )
        filtered, output_sfreq = preprocess_batch(
            values[None, ...],
            sfreq,
            replace(ANNOTATION_VISUALIZATION, normalization="none"),
        )
        local_start = int(round((eeg_start - read_start) * output_sfreq))
        segment = filtered[0, :, local_start : local_start + int(30 * output_sfreq)]
        processed_by_label[label] = segment
        selections.append(
            {
                "label": label,
                "label_name": LABEL_NAMES[label],
                "subject": args.subject,
                "aligned_start_sec": aligned_start,
                "eeg_start_sec": eeg_start,
                "duration_sec": 30,
                "selection_rule": "first_continuous_30s_segment",
                "status": "INFERRED_SEGMENT_SELECTION",
            }
        )

    pd.DataFrame(selections).to_csv(args.source_dir / "fig06_fig07_segments.csv", index=False)
    (args.source_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "confirmed": ANNOTATION_VISUALIZATION.to_dict(),
                "segment_selection": "inferred; paper does not report subject or exact windows",
                "figure07_channel": args.channel,
                "figure07_channel_status": "inferred",
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        args.source_dir / "fig06_waveforms.npz",
        **{f"label_{label}": values for label, values in processed_by_label.items()},
        channels=np.asarray(EDF_CHANNELS),
        sfreq=sfreq,
    )

    figure, axes = plt.subplots(5, 1, figsize=(15, 14), constrained_layout=True)
    time = np.arange(int(30 * sfreq)) / sfreq
    for label, axis in enumerate(axes):
        values_uv = processed_by_label[label] * 1e6
        scale = np.nanpercentile(np.abs(values_uv), 95, axis=1)
        scale = np.maximum(scale, 1e-6)
        display = values_uv / scale[:, None] + np.arange(len(EDF_CHANNELS))[::-1, None] * 2.5
        axis.plot(time, display.T, color="#1F2933", linewidth=0.35)
        axis.set_title(f"{LABEL_NAMES[label]} | Participant {args.subject} | inferred segment")
        axis.set_xlim(0, 30)
        axis.set_yticks([])
        axis.set_ylabel("32 channels")
    axes[-1].set_xlabel("Time (s)")
    figure.savefig(args.output_dir / "figure06_representative_eeg.png", dpi=300)
    figure.savefig(args.output_dir / "figure06_representative_eeg.pdf")
    plt.close(figure)

    channel_index = EDF_CHANNELS.index(args.channel)
    figure, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    band_colors = {
        "delta": "#2D7DD2",
        "theta": "#2A9D8F",
        "alpha": "#F4B942",
        "beta": "#C44536",
    }
    source = {}
    for label, axis in enumerate(axes):
        signal = processed_by_label[label][channel_index]
        for band, (low, high) in BANDS.items():
            filtered = sosfiltfilt(
                butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype="bandpass", output="sos"),
                signal,
            )
            source[f"label_{label}_{band}"] = filtered
            axis.plot(time, filtered * 1e6, color=band_colors[band], linewidth=0.8, label=band)
        axis.set_title(f"{LABEL_NAMES[label]} | {args.channel}")
        axis.set_ylabel("µV")
    axes[0].legend(ncol=4, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    np.savez_compressed(args.source_dir / "fig07_band_waveforms.npz", **source, sfreq=sfreq)
    figure.savefig(args.output_dir / "figure07_band_waveforms.png", dpi=300)
    figure.savefig(args.output_dir / "figure07_band_waveforms.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()

