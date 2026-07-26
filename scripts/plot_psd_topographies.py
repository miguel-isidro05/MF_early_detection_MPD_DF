#!/usr/bin/env python3
"""Generate the Participant 10 Figure 11 PSD-topography reproduction diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from mpd_df.constants import EDF_CHANNELS, LABEL_NAMES
from mpd_df.dataset import EDF_ASSUMED_SCALE_TO_VOLTS, build_alignment, discover_subjects
from mpd_df.preprocessing import (
    ANNOTATION_VISUALIZATION,
    MNE_EEGLAB_LIKE,
    preprocess_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subject", default="10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocessing",
        choices=("annotation_visualization", "mne_eeglab_like"),
        default="annotation_visualization",
    )
    parser.add_argument("--ica-kurtosis-threshold", type=float, default=10.0)
    parser.add_argument("--ica-max-excluded", type=int, default=2)
    parser.add_argument("--average-reference", action="store_true")
    parser.add_argument(
        "--psd-mode",
        choices=("sum_linear", "mean_linear", "db_of_mean", "mean_db"),
        default="sum_linear",
    )
    parser.add_argument(
        "--epoch-aggregation",
        choices=("mean", "median"),
        default="mean",
        help="Mean is the paper-facing default; median is an outlier-sensitivity analysis.",
    )
    parser.add_argument("--psd-low-hz", type=float, default=0.3)
    parser.add_argument("--psd-high-hz", type=float, default=35.0)
    parser.add_argument(
        "--trim-edge-sec",
        type=int,
        default=60,
        help="Figure 6 policy: remove one minute from each recording end.",
    )
    return parser.parse_args()


def apply_clei_ica(
    segments: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    seed: int,
    kurtosis_threshold: float,
    max_excluded: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the deterministic FastICA policy validated in the CLEI benchmark.

    ICA is fitted on a 1 Hz high-passed copy, then applied to the original
    segments.  This is intentionally the same order and parameters as
    ``CLEI_2026/benchmark/src/preprocessing.py``; only its train/test split is
    inapplicable to this descriptive, single-participant figure.
    """

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    epochs_original = mne.EpochsArray(segments.astype(np.float64), info, verbose="ERROR")
    epochs_fit = epochs_original.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    ica = mne.preprocessing.ICA(
        n_components=0.99,
        random_state=seed,
        max_iter="auto",
        method="fastica",
    )
    ica.fit(epochs_fit, verbose="ERROR")
    sources = ica.get_sources(epochs_fit).get_data()
    flattened = np.transpose(sources, (1, 0, 2)).reshape(sources.shape[1], -1)
    kurtosis = stats.kurtosis(flattened, axis=1, fisher=False, bias=False)
    candidates = np.where(np.asarray(kurtosis) > kurtosis_threshold)[0].tolist()
    excluded = sorted(candidates, key=lambda index: float(kurtosis[index]), reverse=True)[
        :max_excluded
    ]
    ica.exclude = [int(index) for index in excluded]
    cleaned = epochs_original.copy()
    ica.apply(cleaned, verbose="ERROR")
    record = {
        "implementation_source": "CLEI_2026/benchmark/src/preprocessing.py::apply_ica_fit_train_apply",
        "fit_filter": {"l_freq_hz": 1.0, "h_freq_hz": None},
        "method": "fastica",
        "n_components": 0.99,
        "random_state": seed,
        "kurtosis_threshold": kurtosis_threshold,
        "max_removed_components": max_excluded,
        "excluded_components": [int(index) for index in excluded],
        "component_kurtosis": [float(value) for value in kurtosis],
    }
    return cleaned.get_data().astype(np.float32), record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.psd_low_hz < args.psd_high_hz:
        raise ValueError("Require 0 <= --psd-low-hz < --psd-high-hz")
    files = next(
        item
        for item in discover_subjects(args.raw_root, include_psg=True)
        if item.subject == args.subject
    )
    alignment = build_alignment(files)
    raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
    raw.pick(list(EDF_CHANNELS))
    sfreq = float(raw.info["sfreq"])
    profiles = {
        "annotation_visualization": ANNOTATION_VISUALIZATION,
        "mne_eeglab_like": MNE_EEGLAB_LIKE,
    }
    config = replace(profiles[args.preprocessing], normalization="none")
    segments_by_label = {label: [] for label in range(5)}
    segment_rows = []
    context = 10

    first_start = args.trim_edge_sec
    last_start = alignment.duration_sec - args.trim_edge_sec - 30
    for start in range(first_start, last_start + 1, 30):
        labels = alignment.labels[start : start + 30]
        if np.unique(labels).size != 1 or int(labels[0]) not in segments_by_label:
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
        if args.average_reference:
            segment = segment - segment.mean(axis=0, keepdims=True)
        segments_by_label[label].append(segment)
        segment_rows.append(
            {
                "subject": args.subject,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "aligned_start_sec": start,
                "duration_sec": 30,
            }
        )

    all_segments = np.concatenate(
        [np.stack(segments_by_label[label]) for label in range(5)], axis=0
    )
    cleaned_segments, ica_record = apply_clei_ica(
        all_segments,
        output_sfreq,
        list(EDF_CHANNELS),
        args.seed,
        args.ica_kurtosis_threshold,
        args.ica_max_excluded,
    )
    powers = {label: [] for label in range(5)}
    offset = 0
    for label in range(5):
        count = len(segments_by_label[label])
        for segment in cleaned_segments[offset : offset + count]:
            frequencies, psd = welch(
                segment,
                fs=output_sfreq,
                nperseg=int(round(2 * output_sfreq)),
                axis=-1,
            )
            # Keep the Figure 6 frequency range for the Figure 11 diagnostic.
            mask = (frequencies >= args.psd_low_hz) & (frequencies <= args.psd_high_hz)
            selected = psd[:, mask]
            if args.psd_mode == "sum_linear":
                power = selected.sum(axis=-1)
            elif args.psd_mode == "mean_linear":
                power = selected.mean(axis=-1)
            elif args.psd_mode == "mean_db":
                power = (
                    10.0 * np.log10(np.maximum(selected * 1e12, np.finfo(float).tiny))
                ).mean(axis=-1)
            else:
                power = selected.mean(axis=-1)
            powers[label].append(power)
        offset += count

    averages = {}
    rows = []
    for label, values in powers.items():
        if not values:
            raise ValueError(f"No PSD segments for label {label}")
        stacked = np.stack(values)
        average = (
            np.mean(stacked, axis=0)
            if args.epoch_aggregation == "mean"
            else np.median(stacked, axis=0)
        )
        if args.psd_mode == "db_of_mean":
            average = 10.0 * np.log10(np.maximum(average * 1e12, np.finfo(float).tiny))
        elif args.psd_mode in {"sum_linear", "mean_linear"}:
            average = average * 1e12
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
                "status": "MNE_ICA_REPRODUCTION_DIAGNOSTIC",
                "limitation": (
                    "The paper does not report its EEGLAB ICA components or rejection criteria; "
                    "the deterministic CLEI FastICA policy is transferred for reproducibility."
                ),
                "preprocessing": config.to_dict(),
                "figure6_edge_trim_sec": args.trim_edge_sec,
                "average_reference": args.average_reference,
                "ica": ica_record,
                "psd": (
                    "Welch, 2-second subwindows, "
                    f"{args.psd_low_hz:g}-{args.psd_high_hz:g} Hz, mode={args.psd_mode}"
                ),
                "epoch_aggregation": args.epoch_aggregation,
                "psd_status": "dB modes are sensitivity analyses; the published 0-5000 scale is linear.",
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
    vmin = 0.0 if args.psd_mode in {"sum_linear", "mean_linear"} else float(all_values.min())
    vmax = float(all_values.max())
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
    unit = "PSD bins (µV²/Hz)" if args.psd_mode == "sum_linear" else "PSD (dB/Hz)" if "db" in args.psd_mode else "Mean PSD (µV²/Hz)"
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label=unit)
    figure.suptitle(
        "Participant 10 | MNE 1 Hz FastICA | "
        f"preproc={args.preprocessing} | PSD={args.psd_mode} | CAR={args.average_reference}"
    )
    figure.savefig(args.output_dir / "figure11_psd_topographies_ica.png", dpi=300)
    figure.savefig(args.output_dir / "figure11_psd_topographies_ica.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
