#!/usr/bin/env python3
"""Generate a clearly labeled no-ICA diagnostic related to Figure 11."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from mpd_df.constants import EDF_CHANNELS, LABEL_NAMES
from mpd_df.dataset import EDF_ASSUMED_SCALE_TO_VOLTS, build_alignment, discover_subjects
from mpd_df.preprocessing import ANNOTATION_VISUALIZATION, preprocess_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subject", default="10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    files = next(
        item
        for item in discover_subjects(args.raw_root, include_psg=True)
        if item.subject == args.subject
    )
    alignment = build_alignment(files)
    raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
    raw.pick(list(EDF_CHANNELS))
    sfreq = float(raw.info["sfreq"])
    config = replace(ANNOTATION_VISUALIZATION, normalization="none")
    powers = {label: [] for label in range(5)}
    segment_rows = []
    context = 10

    for start in range(0, alignment.duration_sec - 30 + 1, 30):
        labels = alignment.labels[start : start + 30]
        if np.unique(labels).size != 1 or int(labels[0]) not in powers:
            continue
        label = int(labels[0])
        eeg_start = alignment.eeg_offset_sec + start
        read_start = max(0, eeg_start - context)
        read_end = min(raw.n_times / sfreq, eeg_start + 30 + context)
        values = (
            raw.get_data(
                start=int(round(read_start * sfreq)),
                stop=int(round(read_end * sfreq)),
            )
            * EDF_ASSUMED_SCALE_TO_VOLTS
        )
        filtered, output_sfreq = preprocess_batch(values[None, ...], sfreq, config)
        local = int(round((eeg_start - read_start) * output_sfreq))
        segment = filtered[0, :, local : local + int(round(30 * output_sfreq))]
        frequencies, psd = welch(
            segment,
            fs=output_sfreq,
            nperseg=int(round(2 * output_sfreq)),
            axis=-1,
        )
        mask = (frequencies >= 1.0) & (frequencies <= 30.0)
        powers[label].append(np.trapezoid(psd[:, mask], frequencies[mask], axis=-1))
        segment_rows.append(
            {
                "subject": args.subject,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "aligned_start_sec": start,
                "duration_sec": 30,
            }
        )

    averages = {}
    rows = []
    for label, values in powers.items():
        if not values:
            raise ValueError(f"No PSD segments for label {label}")
        average = np.mean(np.stack(values), axis=0) * 1e12
        averages[label] = average
        for channel, power in zip(EDF_CHANNELS, average):
            rows.append(
                {
                    "subject": args.subject,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "channel": channel,
                    "power_uv2": float(power),
                    "n_30s_segments": len(values),
                }
            )
    pd.DataFrame(segment_rows).to_csv(args.source_dir / "fig11_segments.csv", index=False)
    pd.DataFrame(rows).to_csv(args.source_dir / "fig11_psd_by_label_channel.csv", index=False)
    np.savez_compressed(
        args.source_dir / "fig11_topomap_values.npz",
        values=np.stack([averages[label] for label in range(5)]),
        labels=np.arange(5),
        channels=np.asarray(EDF_CHANNELS),
    )
    (args.source_dir / "fig11_method.json").write_text(
        json.dumps(
            {
                "status": "NO_ICA_DIAGNOSTIC_NOT_REFERENCE_REPRODUCTION",
                "reason": (
                    "The paper does not report removed ICA components or deterministic "
                    "ocular/muscular rejection criteria."
                ),
                "preprocessing": config.to_dict(),
                "psd": "Welch, 2-second subwindows, integrated 1-30 Hz, converted to uV^2",
                "psd_status": "INFERRED",
                "segment_policy": "non-overlapping aligned 30-second clean single-label segments",
            },
            indent=2,
        )
        + "\n"
    )

    info = mne.create_info(list(EDF_CHANNELS), sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")
    positions = np.asarray([channel["loc"][:2] for channel in info["chs"]])
    all_values = np.concatenate(list(averages.values()))
    vmin, vmax = 0.0, float(all_values.max())
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    image = None
    for label, axis in enumerate(axes.flat[:5]):
        image, _ = mne.viz.plot_topomap(
            averages[label],
            positions,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(vmin, vmax),
            contours=5,
            sensors=True,
        )
        axis.set_title(f"{chr(97 + label)}  {LABEL_NAMES[label]}")
    axes.flat[-1].axis("off")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Integrated PSD (µV²)")
    figure.suptitle("Participant 10 | closest no-ICA diagnostic | inferred PSD method")
    figure.savefig(args.output_dir / "figure11_psd_topographies_no_ica.png", dpi=300)
    figure.savefig(args.output_dir / "figure11_psd_topographies_no_ica.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
