#!/usr/bin/env python3
"""Create a complete publication-facing binary-task result bundle.

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
from scipy.stats import wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from mpd_df.metrics import binary_metrics


EXPERIMENTS = (("within_subject", "psd_svm"), ("within_subject", "random_forest"),
               ("within_subject", "eegnet"), ("loso", "psd_svm"),
               ("loso", "random_forest"), ("loso", "eegnet"))


def eegnet_width(row: pd.Series) -> int:
    if "eegnet_f1" in row.index and pd.notna(row["eegnet_f1"]):
        return int(row["eegnet_f1"])
    key = (round(float(row["dropout"]), 2), round(float(row["learning_rate"]), 4))
    legacy_mapping = {
        (0.25, 0.001): 8,
        (0.50, 0.0003): 8,
        (0.25, 0.0003): 16,
        (0.50, 0.001): 16,
    }
    return legacy_mapping[key]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("A", "B"), required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--channel-ablation-root", type=Path)
    parser.add_argument("--reference-figures-dir", type=Path, default=Path("reproduction/figures"))
    return parser.parse_args()


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.savefig(directory / f"{stem}.png", dpi=300)
    figure.savefig(directory / f"{stem}.pdf")
    plt.close(figure)


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    repetitions: int = 10_000,
) -> tuple[float, float]:
    """Participant bootstrap confidence interval for a subject-level mean."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(repetitions, clean.size), replace=True)
    return tuple(np.quantile(samples.mean(axis=1), [0.025, 0.975]))


def context_aggregation_metrics(
    predictions: pd.DataFrame,
    contexts: tuple[int, ...] = (1, 5, 10, 30),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate adjacent out-of-fold decisions without crossing labels/groups."""

    subject_rows = []
    source_rows = []
    for (protocol, model), experiment in predictions.groupby(
        ["protocol", "model"],
        sort=False,
    ):
        for context in contexts:
            local = experiment.copy()
            group_start = local.groupby(
                ["subject", "group_id"]
            ).window_start_sec.transform("min")
            local["context_bin"] = (
                (local.window_start_sec - group_start) // context
            ).astype(int)
            aggregated = (
                local.groupby(
                    ["subject", "group_id", "context_bin"],
                    as_index=False,
                )
                .agg(
                    target_fraction=("y_true", "mean"),
                    score=("score", "mean"),
                    vote=("y_pred", "mean"),
                    n_windows=("y_pred", "size"),
                )
            )
            aggregated["y_true"] = (
                aggregated.target_fraction >= 0.5
            ).astype(np.int8)
            aggregated["y_pred"] = (aggregated.vote >= 0.5).astype(np.int8)
            aggregated["protocol"] = protocol
            aggregated["model"] = model
            aggregated["context_sec"] = context
            source_rows.append(aggregated)
            for subject, rows in aggregated.groupby("subject"):
                metrics = binary_metrics(
                    rows.y_true.to_numpy(),
                    rows.y_pred.to_numpy(),
                )
                subject_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "context_sec": context,
                        "subject": subject,
                        **{
                            key: metrics[key]
                            for key in (
                                "balanced_accuracy",
                                "precision",
                                "recall",
                                "f1",
                                "kappa",
                            )
                        },
                    }
                )
    return pd.DataFrame(subject_rows), pd.concat(source_rows, ignore_index=True)


def main() -> None:
    parsed = args()
    protocol_config = yaml.safe_load(parsed.protocol_config.read_text())
    if protocol_config.get("task") != parsed.task:
        raise ValueError(
            "Protocol config task does not match the requested result bundle"
        )
    task_slug = f"task_{parsed.task.lower()}"
    positive_label = (
        "Fatigue1" if parsed.task == "A" else "Fatigue1 + Fatigue2"
    )
    positive_definition = (
        "Fatigue1 (1)"
        if parsed.task == "A"
        else "Fatigue1 (1) + Fatigue2 (2)"
    )
    if parsed.output_dir.exists():
        raise FileExistsError(parsed.output_dir)
    tables = parsed.output_dir / "tables"
    figures = parsed.output_dir / "figures"
    tables.mkdir(parents=True)
    figures.mkdir()
    reference_figures = parsed.output_dir / "reference_figures"
    reference_figures.mkdir()
    summary, subjects, predictions, fold_metrics = [], [], [], []
    for experiment_index, (protocol, model) in enumerate(EXPERIMENTS):
        root = parsed.experiments_root / protocol / model
        environment = json.loads((root / "environment.txt").read_text())
        experiment_config = yaml.safe_load((root / "config.yaml").read_text())
        if environment["context"]["task"] != parsed.task:
            raise ValueError(f"{root} contains a different task")
        if experiment_config.get("task") != parsed.task:
            raise ValueError(f"{root}/config.yaml contains a different task")
        metrics = json.loads((root / "metrics.json").read_text())
        macro = metrics["macro_subject"]
        frame = pd.read_csv(root / "subject_metrics.csv")
        summary_row = {
            "protocol": protocol,
            "model": model,
            **{
                f"{key}_mean": macro[key]["mean"]
                for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "kappa")
            },
            **{
                f"{key}_std": macro[key]["std"]
                for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "kappa")
            },
        }
        for metric_name in ("balanced_accuracy", "f1"):
            low, high = bootstrap_mean_ci(
                frame[metric_name].to_numpy(),
                seed=42 + experiment_index,
            )
            summary_row[f"{metric_name}_ci95_low"] = low
            summary_row[f"{metric_name}_ci95_high"] = high
        summary.append(summary_row)
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
                    "eegnet_f1": eegnet_width(row),
                    "learning_rate": row["learning_rate"],
                }, sort_keys=True),
                axis=1,
            )
        fold_metrics.append(frame)
    summary = pd.DataFrame(summary); subjects = pd.concat(subjects, ignore_index=True); predictions = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.concat(fold_metrics, ignore_index=True)
    subject_discrimination_rows = []
    for (protocol, model, subject), group in predictions.groupby(
        ["protocol", "model", "subject"],
        sort=False,
    ):
        if group.y_true.nunique() < 2:
            continue
        subject_discrimination_rows.append({
            "protocol": protocol,
            "model": model,
            "subject": subject,
            "roc_auc": roc_auc_score(group.y_true, group.score),
            "average_precision": average_precision_score(group.y_true, group.score),
        })
    subject_discrimination = pd.DataFrame(subject_discrimination_rows)
    discrimination_macro = (
        subject_discrimination.groupby(["protocol", "model"])[
            ["roc_auc", "average_precision"]
        ]
        .agg(["mean", "std"])
    )
    discrimination_macro.columns = [
        f"{metric}_subject_macro_{statistic}"
        for metric, statistic in discrimination_macro.columns
    ]
    summary = summary.merge(
        discrimination_macro.reset_index(),
        on=["protocol", "model"],
        how="left",
    )
    summary.to_csv(tables / "model_summary.csv", index=False)
    subjects.to_csv(tables / "subject_metrics.csv", index=False)
    subject_discrimination.to_csv(
        tables / "subject_level_discrimination.csv",
        index=False,
    )
    predictions.to_csv(tables / "out_of_fold_predictions.csv", index=False)
    fold_metrics.to_csv(tables / "fold_metrics_and_selection.csv", index=False)
    context_metrics, context_source = context_aggregation_metrics(predictions)
    context_metrics.to_csv(
        tables / "temporal_context_subject_metrics.csv",
        index=False,
    )
    context_source.to_csv(
        tables / "temporal_context_aggregated_predictions.csv",
        index=False,
    )
    context_summary = (
        context_metrics.groupby(["protocol", "model", "context_sec"])[
            ["balanced_accuracy", "f1", "recall"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    context_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in context_summary.columns
    ]
    context_summary.to_csv(
        tables / "temporal_context_summary.csv",
        index=False,
    )
    selection_frequency = (
        fold_metrics.groupby(["protocol", "model", "selected_params"], dropna=False)
        .size().rename("selected_folds").reset_index()
    )
    selection_frequency.to_csv(tables / "selected_hyperparameter_frequency.csv", index=False)
    pd.DataFrame([
        {
            "component": "task",
            "value": (
                f"Task {parsed.task}: Wakefulness (0) versus "
                f"{positive_definition}"
            ),
        },
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
    task_counts = (
        canonical.groupby(["subject", "y_true"]).size().rename("windows").reset_index()
    )
    task_counts.to_csv(tables / f"{task_slug}_label_distribution.csv", index=False)
    count_pivot = (
        task_counts.pivot(index="subject", columns="y_true", values="windows")
        .reindex(columns=[0, 1], fill_value=0)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x_subject = np.arange(len(count_pivot))
    wake = count_pivot[0].to_numpy()
    fatigue = count_pivot[1].to_numpy()
    ax.bar(x_subject, wake, label="Wakefulness", color="#277da1")
    ax.bar(
        x_subject,
        fatigue,
        bottom=wake,
        label=positive_label,
        color="#f94144",
    )
    ax.set(
        xticks=x_subject,
        xticklabels=count_pivot.index.astype(str),
        xlabel="Participant",
        ylabel="Eligible 1-second windows",
        title=f"Task {parsed.task} label distribution after exclusions",
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    save_figure(fig, figures, f"{task_slug}_label_distribution")

    labels = [f"{p.replace('_', ' ')}\n{m.replace('_', ' ').upper()}" for p, m in summary[["protocol", "model"]].itertuples(index=False)]
    x = np.arange(len(summary)); fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(
        x - .18,
        summary.f1_mean,
        .36,
        yerr=summary.f1_std,
        label=f"{positive_label} F1",
        capsize=3,
    )
    ax.bar(x + .18, summary.balanced_accuracy_mean, .36, yerr=summary.balanced_accuracy_std, label="Balanced accuracy", capsize=3)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
        ylabel="Subject-macro score",
        title=(
            f"Task {parsed.task}: personalized and unseen-subject performance"
        ),
    )
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
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Wake", positive_label], yticklabels=["Wake", positive_label],
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

    reliability_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for axis, model in zip(axes, ("random_forest", "eegnet")):
        group = predictions[(predictions.protocol == "loso") & (predictions.model == model)]
        probability = group.score.to_numpy()
        observed, predicted = calibration_curve(group.y_true, probability, n_bins=10, strategy="quantile")
        reliability_rows.append({
            "protocol": "loso",
            "model": model,
            "brier_score_micro_window": brier_score_loss(group.y_true, probability),
        })
        axis.plot([0, 1], [0, 1], "--", color="gray"); axis.plot(predicted, observed, marker="o")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel=f"Observed {positive_label}", title=model)
    pd.DataFrame(reliability_rows).to_csv(
        tables / "loso_probability_reliability.csv",
        index=False,
    )
    save_figure(fig, figures, "loso_probability_reliability")

    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.axis("off")
    labels_flow = ["MPD-DF EEG", "Audit + alignment", "Standard preprocessing", f"Task {parsed.task} windows",
                   "Nested development", "LOSO test", "Subject-level inference"]
    for index, label in enumerate(labels_flow):
        x0 = .02 + index * .14
        ax.text(x0, .5, label, ha="center", va="center",
                bbox={"boxstyle": "round,pad=.35", "facecolor": "#e8f0f7", "edgecolor": "#345"})
        if index < len(labels_flow) - 1:
            ax.annotate("", xy=(x0 + .115, .5), xytext=(x0 + .065, .5), arrowprops={"arrowstyle": "->"})
    save_figure(fig, figures, f"{task_slug}_framework")

    representation = summary[summary.model.isin(["psd_svm", "eegnet"])].copy()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for model, group in representation.groupby("model"):
        ax.plot(group.protocol, group.f1_mean, marker="o", label=model)
    ax.set(ylim=(0, 1), ylabel="Subject-macro F1", title="Representation ablation"); ax.legend(); ax.grid(alpha=.25)
    save_figure(fig, figures, "representation_ablation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, protocol in zip(axes, ("within_subject", "loso")):
        local = context_summary[context_summary.protocol == protocol]
        for model, group in local.groupby("model"):
            axis.plot(
                group.context_sec,
                group.f1_mean,
                marker="o",
                label=model,
            )
        axis.set(
            xticks=[1, 5, 10, 30],
            ylim=(0, 1),
            xlabel="Decision aggregation context (s)",
            ylabel="Subject-macro F1",
            title=protocol.replace("_", " "),
        )
        axis.grid(alpha=0.2)
    axes[0].legend()
    save_figure(fig, figures, "temporal_context_aggregation")

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
        f"{task_slug}_framework.png": (
            f"Leakage-controlled Task {parsed.task} experimental framework."
        ),
        f"{task_slug}_label_distribution.png": (
            f"Eligible Wakefulness and {positive_label} windows by participant."
        ),
        "protocol_comparison.png": "Subject-macro F1 and balanced accuracy by protocol and model.",
        "subject_level_f1_recall.png": "Per-subject out-of-fold F1 and recall.",
        "confusion_matrices_normalized_and_counts.png": "Out-of-fold confusion matrices with row-normalized rates and counts.",
        "roc_curves_out_of_fold.png": "Window-level out-of-fold ROC curves; subject-macro metrics remain primary.",
        "precision_recall_curves_out_of_fold.png": "Window-level out-of-fold precision-recall curves.",
        "loso_probability_reliability.png": "Descriptive LOSO probability reliability for RF and EEGNet; SVM decision scores are excluded because they are not calibrated probabilities.",
        "representation_ablation.png": "PSD versus raw-window representation comparison.",
        "channel_ablation.png": "EDF32 versus paper28 montage comparison.",
        "hyperparameter_selection_frequency.png": "Frequency of nested hyperparameter choices across outer folds.",
        "temporal_context_aggregation.png": (
            "Descriptive aggregation of adjacent out-of-fold decisions at "
            "1, 5, 10, and 30 seconds in fixed clock bins that do not cross "
            "30-second evaluation groups."
        ),
        "eegnet_learning_curves_loso.png": "EEGNet inner-validation learning curves under LOSO.",
        "tsne_psd_exploratory.png": "Exploratory t-SNE of PSD features; not used for inferential claims.",
    }
    pd.DataFrame([
        {"figure": name, "caption": caption}
        for name, caption in captions.items()
    ]).to_csv(tables / "figure_manifest.csv", index=False)
    (parsed.output_dir / "README.md").write_text(
        f"# Final Task {parsed.task} Results\n\n"
        "All predictive metrics and model plots are generated from saved "
        "out-of-fold predictions. Subject-macro metrics are primary; pooled "
        "window ROC/PR curves are descriptive. The t-SNE plot is exploratory.\n"
    )
    audit = parsed.output_dir / "audit"
    audit.mkdir()
    for source in (
        Path(
            "FINAL_TASK_A_PROTOCOL.md"
            if parsed.task == "A"
            else "FINAL_TASK_B_PROTOCOL.md"
        ),
        parsed.protocol_config,
        Path(f"data/metadata/eligible_within_task_{parsed.task.lower()}.txt"),
        Path(f"data/metadata/excluded_within_task_{parsed.task.lower()}.csv"),
        Path("data/metadata/alignment_manifest.csv"),
    ):
        if source.exists():
            shutil.copy2(source, audit / source.name)
    cache_manifests = parsed.experiments_root / "cache_manifests"
    if cache_manifests.exists():
        destination = audit / "cache_manifests"
        destination.mkdir()
        for source in cache_manifests.glob("*.json"):
            shutil.copy2(source, destination / source.name)


if __name__ == "__main__":
    main()
