#!/usr/bin/env python3
"""Compare completed Task A and Task B publication summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-a-results", type=Path, required=True)
    parser.add_argument("--task-b-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for task, root in (("A", args.task_a_results), ("B", args.task_b_results)):
        summary = pd.read_csv(root / "tables" / "model_summary.csv")
        summary.insert(0, "task", task)
        rows.append(summary)
    comparison = pd.concat(rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir.parent / "tables"
    comparison.to_csv(tables / "task_a_b_model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, protocol in zip(axes, ("within_subject", "loso")):
        local = comparison[comparison.protocol == protocol]
        labels = sorted(local.model.unique())
        positions = range(len(labels))
        for offset, task in ((-0.18, "A"), (0.18, "B")):
            values = (
                local[local.task == task]
                .set_index("model")
                .reindex(labels)["f1_mean"]
            )
            axis.bar(
                [value + offset for value in positions],
                values,
                width=0.36,
                label=f"Task {task}",
            )
        axis.set(
            xticks=list(positions),
            xticklabels=[label.replace("_", " ") for label in labels],
            ylim=(0, 1),
            ylabel="Subject-macro F1",
            title=protocol.replace("_", " "),
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Task definition: Fatigue1 versus Fatigue1+Fatigue2")
    fig.savefig(args.output_dir / "task_a_b_comparison.png", dpi=300)
    fig.savefig(args.output_dir / "task_a_b_comparison.pdf")
    plt.close(fig)
    print(f"[task comparison] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
