#!/usr/bin/env python3
"""Assemble the five-subject Task B preprocessing hypothesis screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    rows = []
    for metrics_path in sorted(args.experiments_root.glob("*/metrics.json")):
        root = metrics_path.parent
        metrics = json.loads(metrics_path.read_text())
        config = yaml.safe_load((root / "config.yaml").read_text())
        macro = metrics["macro_subject"]
        rows.append(
            {
                "experiment": root.name,
                "model": config.get("model", "psd_svm"),
                "preprocessing": root.name.split("_w", maxsplit=1)[0],
                "balanced_accuracy": macro["balanced_accuracy"]["mean"],
                "f1": macro["f1"]["mean"],
                "recall": macro["recall"]["mean"],
                "n_subjects": macro["f1"]["n_subjects_defined"],
            }
        )
    if not rows:
        raise FileNotFoundError(args.experiments_root)
    summary = pd.DataFrame(rows).sort_values(["model", "experiment"])
    summary.to_csv(tables / "task_b_hypothesis_summary.csv", index=False)
    for model, group in summary.groupby("model"):
        fig, ax = plt.subplots(
            figsize=(max(8, 0.65 * len(group)), 4.8),
            constrained_layout=True,
        )
        positions = range(len(group))
        ax.bar(
            [value - 0.18 for value in positions],
            group.balanced_accuracy,
            width=0.36,
            label="Balanced accuracy",
        )
        ax.bar(
            [value + 0.18 for value in positions],
            group.f1,
            width=0.36,
            label="F1",
        )
        ax.set(
            xticks=list(positions),
            xticklabels=group.experiment,
            ylim=(0, 1),
            ylabel="Subject-macro score",
            title=f"Task B preprocessing hypotheses: {model}",
        )
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.savefig(
            figures / f"task_b_hypotheses_{model}.png",
            dpi=300,
        )
        fig.savefig(figures / f"task_b_hypotheses_{model}.pdf")
        plt.close(fig)
    (args.output_dir / "README.md").write_text(
        "# Task B Preprocessing Sensitivity Screen\n\n"
        "Five-subject diagnostic only. The primary pipeline was specified "
        "before this screen. These scores are not final estimates and cannot "
        "be used to select or modify the primary method.\n"
    )
    print(f"[hypotheses] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
