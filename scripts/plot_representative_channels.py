#!/usr/bin/env python3
"""Functional EEG-only recreation of the EEG panels in Figure 10."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from mpd_df.dataset import EDF_ASSUMED_SCALE_TO_VOLTS, build_alignment, discover_subjects
from mpd_df.preprocessing import PHYSIOLOGICAL_VALIDATION, preprocess_batch

CHANNELS = ("Fp1", "C3", "T7", "O1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    arrays = []
    selection_rows = []
    output_sfreq = 200.0
    filter_config = replace(PHYSIOLOGICAL_VALIDATION, normalization="none")
    for files in discover_subjects(args.raw_root, include_psg=True):
        alignment = build_alignment(files)
        raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
        raw.pick(list(CHANNELS))
        sfreq = float(raw.info["sfreq"])
        eeg_start = alignment.eeg_offset_sec
        read_end = min(raw.n_times / sfreq, eeg_start + 20)
        values = (
            raw.get_data(
                start=int(round(eeg_start * sfreq)),
                stop=int(round(read_end * sfreq)),
            )
            * EDF_ASSUMED_SCALE_TO_VOLTS
        )
        filtered, output_sfreq = preprocess_batch(values[None, ...], sfreq, filter_config)
        segment = filtered[0, :, : int(round(10 * output_sfreq))]
        mean = segment.mean(axis=-1, keepdims=True)
        std = segment.std(axis=-1, keepdims=True)
        segment = (segment - mean) / np.maximum(std, np.finfo(segment.dtype).eps)
        arrays.append(segment)
        selection_rows.append(
            {
                "subject": files.subject,
                "aligned_start_sec": 0,
                "eeg_start_sec": eeg_start,
                "duration_sec": 10,
                "selection_rule": "first_10s_after_official_alignment",
                "normalization_scope": "per_selected_10s_channel_inferred",
            }
        )
    data = np.stack(arrays)
    pd.DataFrame(selection_rows).to_csv(args.source_dir / "fig10_selection.csv", index=False)
    np.savez_compressed(
        args.source_dir / "fig10_eeg_waveforms.npz",
        data=data,
        subjects=np.asarray([row["subject"] for row in selection_rows]),
        channels=np.asarray(CHANNELS),
        sfreq=output_sfreq,
    )
    (args.source_dir / "fig10_preprocessing.json").write_text(
        json.dumps(
            {
                "confirmed": PHYSIOLOGICAL_VALIDATION.to_dict(),
                "scope_note": (
                    "The paper reports z-score normalization but not its scope. "
                    "This recreation uses each selected 10-second channel."
                ),
                "selection_status": "INFERRED",
                "scope": "EEG panels only; PSG modalities are outside the paper's modeling scope",
            },
            indent=2,
        )
        + "\n"
    )

    time = np.arange(data.shape[-1]) / output_sfreq
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(data)))
    for channel_index, (channel, axis) in enumerate(zip(CHANNELS, axes)):
        for subject_index in range(len(data)):
            axis.plot(
                time,
                data[subject_index, channel_index],
                color=colors[subject_index],
                linewidth=0.45,
                alpha=0.45,
            )
        axis.set_title(channel)
        axis.set_ylabel("z-score")
        axis.set_xlim(0, 10)
    axes[-1].set_xlabel("Time (s)")
    figure.savefig(args.output_dir / "figure10_representative_channels.png", dpi=300)
    figure.savefig(args.output_dir / "figure10_representative_channels.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()

