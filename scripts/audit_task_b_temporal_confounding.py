#!/usr/bin/env python3
"""Measure temporal predictability and adjacent-group sensitivity in Task B."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpd_df.metrics import binary_metrics
from mpd_df.models import make_psd_svm
from mpd_df.splits import within_subject_splits
from run_classical import load_features


def purge_adjacent_groups(
    metadata: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    gap: int,
) -> np.ndarray:
    if gap <= 0:
        return train
    test_blocks = metadata.iloc[test]["block_id"].to_numpy(dtype=int)
    train_blocks = metadata.iloc[train]["block_id"].to_numpy(dtype=int)
    keep = np.ones(len(train), dtype=bool)
    for block in test_blocks:
        keep &= np.abs(train_blocks - block) > gap
    return train[keep]


def nearest_time_prediction(
    metadata: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    train_time = metadata.iloc[train]["window_start_sec"].to_numpy(dtype=float)
    test_time = metadata.iloc[test]["window_start_sec"].to_numpy(dtype=float)
    order = np.argsort(train_time)
    sorted_time = train_time[order]
    positions = np.searchsorted(sorted_time, test_time)
    left = np.clip(positions - 1, 0, len(sorted_time) - 1)
    right = np.clip(positions, 0, len(sorted_time) - 1)
    choose_right = (
        np.abs(sorted_time[right] - test_time)
        < np.abs(sorted_time[left] - test_time)
    )
    nearest = np.where(choose_right, right, left)
    return y[train[order[nearest]]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-gap-groups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y, metadata, _ = load_features(args.feature_dir, "B")
    rows = []
    for subject in metadata.subject.unique():
        indices = np.flatnonzero(metadata.subject.to_numpy() == subject)
        local = metadata.iloc[indices].reset_index(drop=True)
        local_y = y[indices]
        predictions: dict[tuple[str, int], list[np.ndarray]] = {}
        truth: dict[tuple[str, int], list[np.ndarray]] = {}
        for train_local, test_local in within_subject_splits(
            local,
            local_y,
            seed=args.seed,
        ):
            for gap in (0, args.purge_gap_groups):
                train = purge_adjacent_groups(
                    local,
                    train_local,
                    test_local,
                    gap,
                )
                if set(np.unique(local_y[train])) != {0, 1}:
                    continue
                test_truth = local_y[test_local]
                time_pred = nearest_time_prediction(
                    local,
                    local_y,
                    train,
                    test_local,
                )
                model = make_psd_svm(
                    seed=args.seed,
                    c=1.0,
                    max_iter=50_000,
                    tol=1e-3,
                )
                model.fit(X[indices[train]], local_y[train])
                psd_pred = model.predict(X[indices[test_local]])
                for method, predicted in (
                    ("nearest_time", time_pred),
                    ("fixed_psd_svm", psd_pred),
                ):
                    key = (method, gap)
                    predictions.setdefault(key, []).append(predicted)
                    truth.setdefault(key, []).append(test_truth)
        for (method, gap), parts in predictions.items():
            metrics = binary_metrics(
                np.concatenate(truth[(method, gap)]),
                np.concatenate(parts),
            )
            rows.append(
                {
                    "subject": subject,
                    "method": method,
                    "purge_gap_groups": gap,
                    **{
                        name: metrics[name]
                        for name in (
                            "accuracy",
                            "balanced_accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "kappa",
                        )
                    },
                }
            )
    subject_metrics = pd.DataFrame(rows)
    subject_metrics.to_csv(
        args.output_dir / "temporal_sensitivity_subject_metrics.csv",
        index=False,
    )
    summary = (
        subject_metrics.groupby(["method", "purge_gap_groups"])[
            ["balanced_accuracy", "f1", "recall", "kappa"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(
        args.output_dir / "temporal_sensitivity_summary.csv",
        index=False,
    )
    plot = subject_metrics.groupby(
        ["method", "purge_gap_groups"]
    ).balanced_accuracy.mean()
    labels = [
        f"{method}\ngap={gap}"
        for method, gap in plot.index
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(labels, plot.to_numpy())
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set(
        ylim=(0, 1),
        ylabel="Subject-macro balanced accuracy",
        title="Temporal confounding and adjacent-group sensitivity",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(
        args.output_dir / "temporal_sensitivity.png",
        dpi=300,
    )
    fig.savefig(args.output_dir / "temporal_sensitivity.pdf")
    plt.close(fig)
    print(f"[temporal audit] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
