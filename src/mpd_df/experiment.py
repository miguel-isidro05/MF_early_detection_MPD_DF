"""Experiment I/O and classical grouped evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import yaml

from .metrics import binary_metrics


@dataclass(frozen=True)
class ExperimentContext:
    experiment_id: str
    task: str
    model: str
    protocol: str
    seed: int


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNVERSIONED"


def initialize_experiment(
    output_dir: str | Path,
    context: ExperimentContext,
    config: dict[str, object],
    command: str,
    project_root: str | Path,
) -> Path:
    final_output = Path(output_dir)
    if final_output.exists():
        raise FileExistsError(final_output)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f".{final_output.name}.partial-", dir=final_output.parent))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "command.txt").write_text(command.strip() + "\n")
    (output / "seed.txt").write_text(f"{context.seed}\n")
    (output / "git_commit.txt").write_text(git_commit(Path(project_root)) + "\n")
    packages = {}
    for package in ("mne", "numpy", "pandas", "scikit-learn", "scipy", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        import torch

        accelerator = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "mps_available": torch.backends.mps.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        accelerator = None
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "context": asdict(context),
        "packages": packages,
        "accelerator": accelerator,
    }
    (output / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n")
    (output / "stdout.log").touch()
    (output / "stderr.log").touch()
    return output


def finalize_experiment(partial_output: str | Path, final_output: str | Path) -> Path:
    partial = Path(partial_output)
    final = Path(final_output)
    if final.exists():
        raise FileExistsError(final)
    partial.rename(final)
    return final


def evaluate_classical_splits(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    splits: Iterator[tuple[np.ndarray, np.ndarray]],
    estimator_factory: Callable[[], object],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    pooled_predictions = np.full(len(y), -1, dtype=np.int8)
    for fold, (train, test) in enumerate(splits):
        estimator = estimator_factory()
        estimator.fit(X[train], y[train])
        predicted = np.asarray(estimator.predict(X[test]), dtype=np.int8)
        pooled_predictions[test] = predicted
        metrics = binary_metrics(y[test], predicted)
        matrix = metrics.pop("confusion_matrix")
        fold_rows.append({"fold": fold, **metrics, "confusion_matrix": json.dumps(matrix)})
        rows = metadata.iloc[test].copy()
        rows["fold"] = fold
        rows["y_true"] = y[test]
        rows["y_pred"] = predicted
        if hasattr(estimator, "decision_function"):
            rows["score"] = estimator.decision_function(X[test])
        elif hasattr(estimator, "predict_proba"):
            rows["score"] = estimator.predict_proba(X[test])[:, 1]
        prediction_rows.append(rows)
    if np.any(pooled_predictions < 0):
        raise AssertionError("Some samples were not assigned an out-of-fold prediction")
    return pd.DataFrame(fold_rows), pd.concat(prediction_rows, ignore_index=True), pooled_predictions


def save_experiment_results(
    output: str | Path,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    pooled_metrics: dict[str, object],
) -> None:
    root = Path(output)
    fold_metrics.to_csv(root / "fold_metrics.csv", index=False)
    predictions.to_csv(root / "predictions.csv", index=False)
    subject_metrics: list[dict[str, object]] = []
    for subject, rows in predictions.groupby("subject"):
        subject_metrics.append(
            {"subject": subject, **binary_metrics(rows["y_true"].to_numpy(), rows["y_pred"].to_numpy())}
        )
    subject_frame = pd.DataFrame(subject_metrics)
    subject_frame.to_csv(root / "subject_metrics.csv", index=False)
    numeric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "kappa",
    ]
    macro = {}
    for column in numeric_columns:
        values = pd.to_numeric(subject_frame[column], errors="coerce")
        macro[column] = {
            "mean": float(values.mean()) if values.notna().any() else None,
            "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else None,
            "n_subjects_defined": int(values.notna().sum()),
        }
    matrix = np.asarray(pooled_metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]).to_csv(root / "confusion_matrix.csv")
    payload = {
        "micro_window_pooled": pooled_metrics,
        "macro_subject": macro,
        "primary_unit": "subject",
    }
    (root / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
