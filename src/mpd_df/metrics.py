"""Binary fatigue metrics and participant-level aggregation."""

from __future__ import annotations

import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    negative_support = int(true_negative + false_positive)
    positive_support = int(true_positive + false_negative)
    specificity = (
        float(true_negative / negative_support) if negative_support else None
    )
    if np.unique(np.concatenate([y_true, y_pred])).size < 2:
        kappa = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = float(cohen_kappa_score(y_true, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred))
            if negative_support and positive_support
            else None
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": specificity,
        "kappa": kappa if np.isfinite(kappa) else None,
        "confusion_matrix": matrix.tolist(),
        "n": int(len(y_true)),
        "n_negative": negative_support,
        "n_positive": positive_support,
    }
