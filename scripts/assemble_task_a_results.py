#!/usr/bin/env python3
"""Assemble the completed Task A experiments into a reviewable result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = (
    ("within_subject", "psd_svm"),
    ("within_subject", "random_forest"),
    ("within_subject", "eegnet"),
    ("loso", "psd_svm"),
    ("loso", "random_forest"),
    ("loso", "eegnet"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-figures-dir", type=Path, default=Path("reproduction/figures"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_performance_figure(summary: pd.DataFrame, output: Path) -> None:
    labels = [f"{row.protocol}\n{row.model}" for row in summary.itertuples()]
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.bar(x - 0.18, summary["macro_subject_f1_mean"], 0.36, label="Subject-macro F1")
    axis.bar(
        x + 0.18,
        summary["macro_subject_balanced_accuracy_mean"],
        0.36,
        label="Subject-macro balanced accuracy",
    )
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Score")
    axis.set_title("Task A: Wakefulness vs. Fatigue1")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def write_confusion_figure(confusions: list[tuple[str, np.ndarray]], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for axis, (label, matrix) in zip(axes.flat, confusions):
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
        axis.set_title(label)
        axis.set_xticks([0, 1], ["Wake", "Fatigue1"])
        axis.set_yticks([0, 1], ["Wake", "Fatigue1"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Windows")
    figure.suptitle("Task A confusion matrices")
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    sources = args.output_dir / "reference_figures"
    tables.mkdir(parents=True)
    figures.mkdir()
    sources.mkdir()

    rows = []
    subject_frames = []
    confusions = []
    manifest = {"task": "A", "experiments": []}
    for protocol, model in EXPERIMENTS:
        root = args.experiments_root / protocol / f"A_1s_{model}"
        metrics_path = root / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text())
        macro = metrics["macro_subject"]
        row = {
            "protocol": protocol,
            "model": model,
            "primary_unit": metrics["primary_unit"],
            "n_subjects": macro["f1"]["n_subjects_defined"],
        }
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "kappa"):
            row[f"macro_subject_{metric}_mean"] = macro[metric]["mean"]
            row[f"macro_subject_{metric}_std"] = macro[metric]["std"]
        rows.append(row)

        subject_metrics = pd.read_csv(root / "subject_metrics.csv")
        subject_metrics.insert(0, "model", model)
        subject_metrics.insert(0, "protocol", protocol)
        subject_frames.append(subject_metrics)
        matrix = pd.read_csv(root / "confusion_matrix.csv", index_col=0).to_numpy()
        confusions.append((f"{protocol}\n{model}", matrix))
        manifest["experiments"].append(
            {
                "protocol": protocol,
                "model": model,
                "path": str(root),
                "metrics_sha256": sha256(metrics_path),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "task_a_model_summary.csv", index=False)
    pd.concat(subject_frames, ignore_index=True).to_csv(
        tables / "task_a_subject_metrics.csv", index=False
    )
    write_performance_figure(summary, figures / "task_a_model_comparison")
    write_confusion_figure(confusions, figures / "task_a_confusion_matrices")

    for path in sorted(args.reference_figures_dir.glob("*.png")):
        shutil.copy2(path, sources / path.name)
    for path in sorted(args.reference_figures_dir.glob("*.pdf")):
        shutil.copy2(path, sources / path.name)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Task A Result Bundle\n\n"
        "Primary comparison: subject-macro F1 and balanced accuracy.\n\n"
        "`tables/task_a_model_summary.csv` contains the six final comparisons.\n"
        "`figures/` contains model-comparison and confusion-matrix figures.\n"
        "`reference_figures/` contains the separately reproduced dataset figures.\n"
        "The raw experiment folders remain under `experiments/`.\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
