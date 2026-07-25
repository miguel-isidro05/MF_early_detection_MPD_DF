#!/usr/bin/env python3
"""Recreate the MPD-DF label timeline and participant distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from mpd_df.constants import LABEL_NAMES
from mpd_df.dataset import build_alignment, discover_subjects

LABEL_ORDER = [0, 1, 2, 3, 4, 8, 9]
COLORS = ["#2D7DD2", "#F4B942", "#E67E22", "#C44536", "#7D3C98", "#6C757D", "#111111"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


def run_length_rows(subject: str, labels: np.ndarray) -> list[dict[str, int | str]]:
    boundaries = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        label = int(labels[start])
        rows.append(
            {
                "subject": subject,
                "start_sec": int(start),
                "end_sec": int(end),
                "duration_sec": int(end - start),
                "label": label,
                "label_name": LABEL_NAMES[label],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    aligned = []
    segment_rows = []
    count_rows = []
    for files in discover_subjects(args.raw_root, include_psg=True):
        result = build_alignment(files)
        aligned.append((files.subject, result.labels))
        segment_rows.extend(run_length_rows(files.subject, result.labels))
        normalized = result.labels[60 : min(len(result.labels), 60 + 118 * 60)]
        for label in LABEL_ORDER:
            count_rows.append(
                {
                    "subject": files.subject,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "normalized_118min_seconds": int(np.count_nonzero(normalized == label)),
                }
            )

    pd.DataFrame(segment_rows).to_csv(args.source_dir / "label_segments.csv", index=False)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(args.source_dir / "label_counts_normalized_118min.csv", index=False)

    maximum = max(len(labels) for _, labels in aligned)
    matrix = np.full((len(aligned), maximum), np.nan, dtype=float)
    encoded = {label: index for index, label in enumerate(LABEL_ORDER)}
    for row, (_, labels) in enumerate(aligned):
        matrix[row, : len(labels)] = [encoded[int(label)] for label in labels]
    np.savez_compressed(
        args.source_dir / "label_timeline_matrix.npz",
        matrix=matrix,
        subjects=np.asarray([subject for subject, _ in aligned]),
        label_order=np.asarray(LABEL_ORDER),
    )

    cmap = ListedColormap(COLORS)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, len(LABEL_ORDER) + 0.5), cmap.N)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )
    image = axes[0].imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    axes[0].set_xlabel("Aligned time (s)")
    axes[0].set_ylabel("Participant")
    axes[0].set_yticks(np.arange(0, 50, 5), [f"{index + 1:02d}" for index in range(0, 50, 5)])
    colorbar = figure.colorbar(image, ax=axes[0], ticks=range(len(LABEL_ORDER)), pad=0.01)
    colorbar.ax.set_yticklabels([LABEL_NAMES[label] for label in LABEL_ORDER])

    pivot = counts.pivot(index="subject", columns="label", values="normalized_118min_seconds")
    pivot = pivot.reindex(columns=LABEL_ORDER, fill_value=0)
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for label, color in zip(LABEL_ORDER, COLORS):
        values = pivot[label].to_numpy()
        axes[1].bar(x, values, bottom=bottom, color=color, width=0.85, label=LABEL_NAMES[label])
        bottom += values
    axes[1].set_xlabel("Participant")
    axes[1].set_ylabel("Normalized duration (s)")
    axes[1].set_xticks(np.arange(0, 50, 5), [f"{index + 1:02d}" for index in range(0, 50, 5)])
    axes[1].legend(ncol=4, frameon=False, fontsize=8, loc="upper center")
    figure.savefig(args.figure_dir / "label_distribution.png", dpi=300)
    figure.savefig(args.figure_dir / "label_distribution.pdf")
    plt.close(figure)

    zoom_start, zoom_end = 6900, 7350
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    zoom = matrix[:, zoom_start:zoom_end]
    axis.imshow(
        zoom,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(zoom_start, zoom_end, len(aligned) + 0.5, 0.5),
    )
    axis.axvline(7200, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Aligned time (s)")
    axis.set_ylabel("Participant")
    axis.set_yticks(np.arange(1, 51, 5))
    figure.savefig(args.figure_dir / "label_distribution_7200s_zoom.png", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()

