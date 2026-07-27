#!/usr/bin/env python3
"""Create the complete publication-facing Task A result bundle.

The script consumes experiment folders; it never retrains or recomputes a
metric from in-sample predictions. All plots are derived from saved out-of-fold
predictions and learning curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


EXPERIMENTS = (("within_subject", "psd_svm"), ("within_subject", "random_forest"),
               ("within_subject", "eegnet"), ("loso", "psd_svm"),
               ("loso", "random_forest"), ("loso", "eegnet"))


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments/final_task_a"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-ablation-root", type=Path)
    parser.add_argument("--reference-figures-dir", type=Path, default=Path("reproduction/figures"))
    return parser.parse_args()


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.savefig(directory / f"{stem}.png", dpi=300)
    figure.savefig(directory / f"{stem}.pdf")
    plt.close(figure)


def main() -> None:
    parsed = args()
    if parsed.output_dir.exists():
        raise FileExistsError(parsed.output_dir)
    tables = parsed.output_dir / "tables"
    figures = parsed.output_dir / "figures"
    tables.mkdir(parents=True)
    figures.mkdir()
    reference_figures = parsed.output_dir / "reference_figures"
    reference_figures.mkdir()
    summary, subjects, predictions, fold_metrics = [], [], [], []
    for protocol, model in EXPERIMENTS:
        root = parsed.experiments_root / protocol / model
        metrics = json.loads((root / "metrics.json").read_text())
        macro = metrics["macro_subject"]
        summary.append({"protocol": protocol, "model": model,
                        **{f"{key}_mean": macro[key]["mean"] for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "kappa")},
                        **{f"{key}_std": macro[key]["std"] for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "kappa")}})
        frame = pd.read_csv(root / "subject_metrics.csv")
        frame.insert(0, "model", model); frame.insert(0, "protocol", protocol)
        subjects.append(frame)
        frame = pd.read_csv(root / "predictions.csv")
        frame.insert(0, "model", model); frame.insert(0, "protocol", protocol)
        predictions.append(frame)
        frame = pd.read_csv(root / "fold_metrics.csv")
        frame.insert(0, "model", model); frame.insert(0, "protocol", protocol)
        if model == "eegnet":
            frame["selected_params"] = frame.apply(
                lambda row: json.dumps({
                    "dropout": row["dropout"],
                    "f1": int(row["f1"]),
                    "learning_rate": row["learning_rate"],
                }, sort_keys=True),
                axis=1,
            )
        fold_metrics.append(frame)
    summary = pd.DataFrame(summary); subjects = pd.concat(subjects, ignore_index=True); predictions = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.concat(fold_metrics, ignore_index=True)
    summary.to_csv(tables / "model_summary.csv", index=False)
    subjects.to_csv(tables / "subject_metrics.csv", index=False)
    predictions.to_csv(tables / "out_of_fold_predictions.csv", index=False)
    fold_metrics.to_csv(tables / "fold_metrics_and_selection.csv", index=False)
    selection_frequency = (
        fold_metrics.groupby(["protocol", "model", "selected_params"], dropna=False)
        .size().rename("selected_folds").reset_index()
    )
    selection_frequency.to_csv(tables / "selected_hyperparameter_frequency.csv", index=False)
    pd.DataFrame([
        {"component": "task", "value": "Task A: Wakefulness (0) versus Fatigue1 (1)"},
        {"component": "window", "value": "1 second, 1-second stride, annotation-bin bounded"},
        {"component": "bandpass", "value": "1-100 Hz, zero phase"},
        {"component": "notch", "value": "50 Hz, zero phase"},
        {"component": "centering", "value": "channel mean removal"},
        {"component": "sampling_rate", "value": "200 Hz"},
        {"component": "EEGNet normalization", "value": "per-window/per-channel z-score"},
        {"component": "PSD normalization", "value": "none before PSD; training-fold StandardScaler"},
        {"component": "primary protocol", "value": "nested LOSO by participant"},
        {"component": "development protocol", "value": "nested grouped within-subject"},
        {"component": "seed", "value": "42"},
    ]).to_csv(tables / "protocol_preprocessing.csv", index=False)

    statistic_rows = []
    for protocol in ("within_subject", "loso"):
        local = subjects[subjects.protocol == protocol]
        for left, right in (
            ("psd_svm", "eegnet"),
            ("psd_svm", "random_forest"),
            ("eegnet", "random_forest"),
        ):
            paired = local[local.model == left][["subject", "f1"]].merge(
                local[local.model == right][["subject", "f1"]], on="subject",
                suffixes=("_left", "_right"),
            ).dropna()
            if len(paired) and np.any(paired.f1_left != paired.f1_right):
                result = wilcoxon(paired.f1_left, paired.f1_right, alternative="two-sided")
                statistic_rows.append({
                    "protocol": protocol, "metric": "f1", "left": left, "right": right,
                    "n": len(paired), "statistic": result.statistic, "p_value": result.pvalue,
                    "median_paired_difference": float(
                        np.median(paired.f1_left - paired.f1_right)
                    ),
                })
    statistics = pd.DataFrame(statistic_rows)
    if len(statistics):
        statistics["p_value_holm"] = np.nan
        for _, indices in statistics.groupby("protocol").groups.items():
            ordered = statistics.loc[indices].sort_values("p_value").index
            adjusted = np.maximum.accumulate([
                min(1.0, statistics.loc[index, "p_value"] * (len(ordered) - rank))
                for rank, index in enumerate(ordered)
            ])
            statistics.loc[ordered, "p_value_holm"] = adjusted
    statistics.to_csv(tables / "wilcoxon_pairwise.csv", index=False)

    canonical = predictions[
        (predictions.protocol == "loso") & (predictions.model == "psd_svm")
    ]
    task_a_counts = (
        canonical.groupby(["subject", "y_true"]).size().rename("windows").reset_index()
    )
    task_a_counts.to_csv(tables / "task_a_label_distribution.csv", index=False)
    count_pivot = (
        task_a_counts.pivot(index="subject", columns="y_true", values="windows")
        .reindex(columns=[0, 1], fill_value=0)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x_subject = np.arange(len(count_pivot))
    wake = count_pivot[0].to_numpy()
    fatigue = count_pivot[1].to_numpy()
    ax.bar(x_subject, wake, label="Wakefulness", color="#277da1")
    ax.bar(x_subject, fatigue, bottom=wake, label="Fatigue1", color="#f94144")
    ax.set(
        xticks=x_subject,
        xticklabels=count_pivot.index.astype(str),
        xlabel="Participant",
        ylabel="Eligible 1-second windows",
        title="Task A label distribution after exclusions",
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    save_figure(fig, figures, "task_a_label_distribution")

    labels = [f"{p.replace('_', ' ')}\n{m.replace('_', ' ').upper()}" for p, m in summary[["protocol", "model"]].itertuples(index=False)]
    x = np.arange(len(summary)); fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - .18, summary.f1_mean, .36, yerr=summary.f1_std, label="Fatigue1 F1", capsize=3)
    ax.bar(x + .18, summary.balanced_accuracy_mean, .36, yerr=summary.balanced_accuracy_std, label="Balanced accuracy", capsize=3)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Subject-macro score", title="Task A: personalized and unseen-subject performance")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    save_figure(fig, figures, "protocol_comparison")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, constrained_layout=True)
    for axis, ((protocol, model), group) in zip(axes.flat, subjects.groupby(["protocol", "model"], sort=False)):
        group = group.sort_values("subject")
        axis.plot(group.subject.astype(str), group.f1, marker="o", label="F1")
        axis.plot(group.subject.astype(str), group.recall, marker=".", label="Recall")
        axis.set_title(f"{protocol}: {model}"); axis.tick_params(axis="x", rotation=90, labelsize=7); axis.set_ylim(0, 1); axis.grid(alpha=.2)
    axes[0, 0].legend()
    save_figure(fig, figures, "subject_level_f1_recall")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for axis, ((protocol, model), group) in zip(axes.flat, predictions.groupby(["protocol", "model"], sort=False)):
        matrix = pd.crosstab(group.y_true, group.y_pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{normalized[row, column]:.2f}\n(n={matrix[row, column]})", ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Wake", "Fatigue1"], yticklabels=["Wake", "Fatigue1"],
                 xlabel="Predicted", ylabel="True", title=f"{protocol}: {model}")
    save_figure(fig, figures, "confusion_matrices_normalized_and_counts")

    discrimination_rows = []
    fig_roc, axes_roc = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    fig_pr, axes_pr = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    grouped_predictions = predictions.groupby(["protocol", "model"], sort=False)
    for roc_axis, pr_axis, ((protocol, model), group) in zip(
        axes_roc.flat, axes_pr.flat, grouped_predictions
    ):
        false_positive, true_positive, _ = roc_curve(group.y_true, group.score)
        precision, recall, _ = precision_recall_curve(group.y_true, group.score)
        roc_auc = auc(false_positive, true_positive)
        average_precision = average_precision_score(group.y_true, group.score)
        discrimination_rows.append({
            "protocol": protocol,
            "model": model,
            "roc_auc_micro_window": roc_auc,
            "average_precision_micro_window": average_precision,
        })
        roc_axis.plot(false_positive, true_positive, label=f"AUC={roc_auc:.3f}")
        roc_axis.plot([0, 1], [0, 1], "--", color="gray")
        roc_axis.set(
            xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate",
            ylabel="True-positive rate", title=f"{protocol}: {model}",
        )
        roc_axis.legend()
        prevalence = float(group.y_true.mean())
        pr_axis.plot(recall, precision, label=f"AP={average_precision:.3f}")
        pr_axis.axhline(prevalence, linestyle="--", color="gray")
        pr_axis.set(
            xlim=(0, 1), ylim=(0, 1), xlabel="Recall",
            ylabel="Precision", title=f"{protocol}: {model}",
        )
        pr_axis.legend()
    pd.DataFrame(discrimination_rows).to_csv(
        tables / "window_level_discrimination.csv", index=False
    )
    save_figure(fig_roc, figures, "roc_curves_out_of_fold")
    save_figure(fig_pr, figures, "precision_recall_curves_out_of_fold")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for axis, model in zip(axes, ("psd_svm", "random_forest", "eegnet")):
        group = predictions[(predictions.protocol == "loso") & (predictions.model == model)]
        probability = group.score.to_numpy()
        if model == "psd_svm":
            probability = expit(probability)
        observed, predicted = calibration_curve(group.y_true, probability, n_bins=10, strategy="quantile")
        axis.plot([0, 1], [0, 1], "--", color="gray"); axis.plot(predicted, observed, marker="o")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted score", ylabel="Observed Fatigue1", title=model)
    save_figure(fig, figures, "loso_calibration_curves")

    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.axis("off")
    labels_flow = ["MPD-DF EEG", "Audit + alignment", "Standard preprocessing", "Task A windows",
                   "Nested development", "LOSO test", "Subject-level inference"]
    for index, label in enumerate(labels_flow):
        x0 = .02 + index * .14
        ax.text(x0, .5, label, ha="center", va="center",
                bbox={"boxstyle": "round,pad=.35", "facecolor": "#e8f0f7", "edgecolor": "#345"})
        if index < len(labels_flow) - 1:
            ax.annotate("", xy=(x0 + .115, .5), xytext=(x0 + .065, .5), arrowprops={"arrowstyle": "->"})
    save_figure(fig, figures, "task_a_framework")

    representation = summary[summary.model.isin(["psd_svm", "eegnet"])].copy()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for model, group in representation.groupby("model"):
        ax.plot(group.protocol, group.f1_mean, marker="o", label=model)
    ax.set(ylim=(0, 1), ylabel="Subject-macro F1", title="Representation ablation"); ax.legend(); ax.grid(alpha=.25)
    save_figure(fig, figures, "representation_ablation")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, ((protocol, model), group) in zip(
        axes.flat, selection_frequency.groupby(["protocol", "model"], sort=False)
    ):
        labels = [f"H{index + 1}" for index in range(len(group))]
        axis.bar(labels, group.selected_folds)
        axis.set(
            ylabel="Outer folds selected",
            title=f"{protocol}: {model}",
        )
        axis.grid(axis="y", alpha=.2)
    save_figure(fig, figures, "hyperparameter_selection_frequency")

    if parsed.channel_ablation_root:
        channel_rows = []
        for model in ("psd_svm", "eegnet"):
            path = parsed.channel_ablation_root / "loso" / model / "metrics.json"
            if path.exists():
                payload = json.loads(path.read_text())
                channel_rows.append({"montage": "paper28", "model": model, "f1": payload["macro_subject"]["f1"]["mean"]})
                base = summary[(summary.protocol == "loso") & (summary.model == model)].iloc[0]
                channel_rows.append({"montage": "edf32", "model": model, "f1": base.f1_mean})
        channel = pd.DataFrame(channel_rows)
        channel.to_csv(tables / "channel_ablation.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for model, group in channel.groupby("model"):
            ax.plot(group.montage, group.f1, marker="o", label=model)
        ax.set(ylim=(0, 1), ylabel="Subject-macro F1", title="Channel montage ablation"); ax.legend(); ax.grid(alpha=.25)
        save_figure(fig, figures, "channel_ablation")

    if parsed.reference_figures_dir.exists():
        for pattern in ("*.png", "*.pdf"):
            for path in parsed.reference_figures_dir.glob(pattern):
                shutil.copy2(path, reference_figures / path.name)

    eegnet_root = parsed.experiments_root / "loso" / "eegnet" / "learning_curves.csv"
    if eegnet_root.exists():
        curves = pd.read_csv(eegnet_root)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        mean = curves.groupby("epoch")[["train_loss", "validation_loss"]].mean()
        ax.plot(mean.index, mean.train_loss, label="Train loss"); ax.plot(mean.index, mean.validation_loss, label="Validation loss")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy", title="EEGNet LOSO inner-validation learning curves"); ax.legend(); ax.grid(alpha=.25)
        save_figure(fig, figures, "eegnet_learning_curves_loso")

    captions = {
        "task_a_framework.png": "Leakage-controlled Task A experimental framework.",
        "task_a_label_distribution.png": "Eligible Wakefulness and Fatigue1 windows by participant.",
        "protocol_comparison.png": "Subject-macro F1 and balanced accuracy by protocol and model.",
        "subject_level_f1_recall.png": "Per-subject out-of-fold F1 and recall.",
        "confusion_matrices_normalized_and_counts.png": "Out-of-fold confusion matrices with row-normalized rates and counts.",
        "roc_curves_out_of_fold.png": "Window-level out-of-fold ROC curves; subject-macro metrics remain primary.",
        "precision_recall_curves_out_of_fold.png": "Window-level out-of-fold precision-recall curves.",
        "loso_calibration_curves.png": "LOSO score reliability diagnostics.",
        "representation_ablation.png": "PSD versus raw-window representation comparison.",
        "channel_ablation.png": "EDF32 versus paper28 montage comparison.",
        "hyperparameter_selection_frequency.png": "Frequency of nested hyperparameter choices across outer folds.",
        "eegnet_learning_curves_loso.png": "EEGNet inner-validation learning curves under LOSO.",
        "tsne_psd_exploratory.png": "Exploratory t-SNE of PSD features; not used for inferential claims.",
    }
    pd.DataFrame([
        {"figure": name, "caption": caption}
        for name, caption in captions.items()
    ]).to_csv(tables / "figure_manifest.csv", index=False)
    (parsed.output_dir / "README.md").write_text(
        "# Final Task A Results\n\n"
        "All predictive metrics and model plots are generated from saved "
        "out-of-fold predictions. Subject-macro metrics are primary; pooled "
        "window ROC/PR curves are descriptive. The t-SNE plot is exploratory.\n"
    )


if __name__ == "__main__":
    main()
