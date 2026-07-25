#!/usr/bin/env python3
"""Closest data-grounded EEG-only recreations of MPD-DF Figures 6 and 7."""

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

FIGURE6_SUBJECTS = {0: "10", 1: "10", 2: "10", 3: "10", 4: "10", 8: "30", 9: "05"}
# The x-axis starts are visible in the published Figure 7. Subject identities are
# not reported; these deterministic candidates match the displayed label/time.
FIGURE7_SELECTIONS = {
    0: ("10", 1740),
    1: ("10", 7103),
    2: ("04", 3900),
    3: ("10", 7754),
    4: ("10", 8027),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--channel", default="O1")
    return parser.parse_args()


def first_full_segment(labels: np.ndarray, label: int, duration: int = 30) -> int:
    run = np.convolve((labels == label).astype(int), np.ones(duration, dtype=int), mode="valid")
    matches = np.flatnonzero(run == duration)
    if not matches.size:
        raise ValueError(f"No {duration}s segment for label {label}")
    return int(matches[0])


def extract_segment(files, label: int, raw_start_sec: int | None = None):
    alignment = build_alignment(files)
    aligned_start = (
        first_full_segment(alignment.labels, label)
        if raw_start_sec is None
        else int(raw_start_sec - alignment.eeg_offset_sec)
    )
    selected_labels = alignment.labels[aligned_start : aligned_start + 30]
    if len(selected_labels) != 30 or np.unique(selected_labels).tolist() != [label]:
        raise ValueError(
            f"Subject {files.subject}, raw second {raw_start_sec}: not a clean label {label} segment"
        )
    raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
    raw.pick(list(EDF_CHANNELS))
    sfreq = float(raw.info["sfreq"])
    eeg_start = alignment.eeg_offset_sec + aligned_start
    context_sec = 10
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
    segment = filtered[0, :, local_start : local_start + int(round(30 * output_sfreq))]
    return segment, output_sfreq, aligned_start, eeg_start


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    files_by_subject = {
        files.subject: files for files in discover_subjects(args.raw_root, include_psg=True)
    }
    rows = []
    figure6 = {}
    for label, subject in FIGURE6_SUBJECTS.items():
        segment, sfreq, aligned_start, eeg_start = extract_segment(
            files_by_subject[subject], label
        )
        figure6[label] = segment
        rows.append(
            {
                "figure": 6,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "subject": subject,
                "aligned_start_sec": aligned_start,
                "eeg_start_sec": eeg_start,
                "duration_sec": 30,
                "selection_rule": "first_continuous_30s_candidate",
                "status": "INFERRED_SUBJECT_AND_WINDOW",
            }
        )

    figure7 = {}
    for label, (subject, raw_start) in FIGURE7_SELECTIONS.items():
        segment, sfreq, aligned_start, eeg_start = extract_segment(
            files_by_subject[subject], label, raw_start_sec=raw_start
        )
        figure7[label] = segment
        rows.append(
            {
                "figure": 7,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "subject": subject,
                "aligned_start_sec": aligned_start,
                "eeg_start_sec": eeg_start,
                "duration_sec": 30,
                "selection_rule": "published_x_axis_time_with_matching_subject_candidate",
                "status": "PAPER_TIME_VERIFIED_SUBJECT_INFERRED",
            }
        )
    pd.DataFrame(rows).to_csv(args.source_dir / "fig06_fig07_segments.csv", index=False)
    (args.source_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "confirmed": ANNOTATION_VISUALIZATION.to_dict(),
                "figure6": "all seven published labels; subjects/windows remain inferred",
                "figure7": "published x-axis starts used; subjects remain inferred",
                "figure07_channel": args.channel,
                "figure07_channel_status": "inferred because the paper does not identify it",
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        args.source_dir / "fig06_waveforms.npz",
        **{f"label_{label}": values for label, values in figure6.items()},
        channels=np.asarray(EDF_CHANNELS),
        sfreq=sfreq,
    )

    labels = [0, 1, 2, 3, 4, 8, 9]
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    time = np.arange(int(30 * sfreq)) / sfreq
    for panel, (label, axis) in enumerate(zip(labels, axes.flat)):
        values_uv = figure6[label] * 1e6
        channel_scale = np.maximum(np.nanpercentile(np.abs(values_uv), 95, axis=1), 1e-6)
        display = values_uv / channel_scale[:, None] + np.arange(len(EDF_CHANNELS))[::-1, None] * 2.4
        axis.plot(time, display.T, color="#173B73", linewidth=0.3)
        axis.set_title(f"{chr(97 + panel)}  {LABEL_NAMES[label]}")
        axis.set_xlim(0, 30)
        axis.set_yticks(np.arange(len(EDF_CHANNELS))[::-1] * 2.4)
        axis.set_yticklabels(EDF_CHANNELS, fontsize=5)
        axis.set_xlabel("Time (s)")
    axes.flat[-1].axis("off")
    fig.savefig(args.output_dir / "figure06_representative_eeg.png", dpi=300)
    fig.savefig(args.output_dir / "figure06_representative_eeg.pdf")
    plt.close(fig)

    channel_index = EDF_CHANNELS.index(args.channel)
    band_order = ["beta", "alpha", "theta", "delta"]
    band_colors = {
        "beta": "#E31A1C",
        "alpha": "#16851B",
        "theta": "#18BFC4",
        "delta": "#2447E5",
    }
    source = {}
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for panel, (label, axis) in enumerate(zip(range(5), axes.flat)):
        signal = figure7[label][channel_index]
        waveforms = {}
        for band in band_order:
            low, high = BANDS[band]
            waveforms[band] = sosfiltfilt(
                butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype="bandpass", output="sos"),
                signal,
            )
            source[f"label_{label}_{band}"] = waveforms[band]
        scale = max(
            np.nanpercentile(np.abs(np.concatenate(list(waveforms.values()))) * 1e6, 98),
            1e-6,
        )
        raw_start = FIGURE7_SELECTIONS[label][1]
        panel_time = raw_start + time
        for offset, band in zip([3.0, 2.0, 1.0, 0.0], band_order):
            axis.plot(
                panel_time,
                waveforms[band] * 1e6 / scale * 0.42 + offset,
                color=band_colors[band],
                linewidth=0.7,
            )
        axis.set_yticks([3, 2, 1, 0], ["β", "α", "θ", "δ"])
        axis.set_title(f"{chr(97 + panel)}  {LABEL_NAMES[label]}")
        axis.set_xlim(raw_start, raw_start + 30)
        axis.set_xlabel("Time (s)")
    axes.flat[-1].axis("off")
    np.savez_compressed(args.source_dir / "fig07_band_waveforms.npz", **source, sfreq=sfreq)
    fig.savefig(args.output_dir / "figure07_band_waveforms.png", dpi=300)
    fig.savefig(args.output_dir / "figure07_band_waveforms.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
