#!/usr/bin/env python3
"""Recompute all EEG label-distribution rows reported in MPD-DF Table 7."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mpd_df.dataset import build_alignment, discover_subjects

PAPER = {
    ("1s", 1): [266718, 60292, 14789, 760, 0],
    ("10s_no_overlap", 10): [26716, 5994, 1469, 75, 0],
    ("10s_5s_overlap", 5): [53403, 11968, 2936, 152, 0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_labels(labels: np.ndarray) -> np.ndarray:
    return labels[60 : min(len(labels), 60 + 118 * 60)]


def count_windows(labels: np.ndarray, window: int, stride: int) -> np.ndarray:
    counts = np.zeros(5, dtype=int)
    for start in range(0, len(labels) - window + 1, stride):
        values = labels[start : start + window]
        if np.unique(values).size == 1 and 0 <= int(values[0]) <= 4:
            counts[int(values[0])] += 1
    return counts


def main() -> None:
    args = parse_args()
    sequences = [
        normalized_labels(build_alignment(files).labels)
        for files in discover_subjects(args.raw_root, include_psg=True)
    ]
    rows = []
    for (strategy, stride), reference in PAPER.items():
        window = 1 if strategy == "1s" else 10
        observed = sum((count_windows(labels, window, stride) for labels in sequences), start=np.zeros(5, dtype=int))
        for label in range(5):
            rows.append(
                {
                    "strategy": strategy,
                    "window_sec": window,
                    "stride_sec": stride,
                    "label": label,
                    "observed": int(observed[label]),
                    "paper": reference[label],
                    "difference": int(observed[label] - reference[label]),
                }
            )
        rows.extend(
            [
                {
                    "strategy": strategy,
                    "window_sec": window,
                    "stride_sec": stride,
                    "label": "binary_wakefulness",
                    "observed": int(observed[0]),
                    "paper": reference[0],
                    "difference": int(observed[0] - reference[0]),
                },
                {
                    "strategy": strategy,
                    "window_sec": window,
                    "stride_sec": stride,
                    "label": "binary_fatigue",
                    "observed": int(observed[1:].sum()),
                    "paper": sum(reference[1:]),
                    "difference": int(observed[1:].sum() - sum(reference[1:])),
                },
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
