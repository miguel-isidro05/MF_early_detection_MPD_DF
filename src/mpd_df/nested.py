"""Reusable inner-validation selection for leakage-safe binary experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .metrics import binary_metrics


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Return stable score cutoffs without using outer-test labels."""
    unique = np.unique(np.asarray(scores, dtype=float))
    if unique.size <= 64:
        return unique
    return np.quantile(unique, np.linspace(0.02, 0.98, 49))


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, object]]:
    """Select by fatigue F1, then recall, kappa and balanced accuracy."""
    best: tuple[tuple[float, float, float, float], float, dict[str, object]] | None = None
    for threshold in threshold_candidates(scores):
        metrics = binary_metrics(y_true, np.asarray(scores >= threshold, dtype=np.int8))
        rank = tuple(float(metrics.get(name) or -1.0) for name in ("f1", "recall", "kappa", "balanced_accuracy"))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    if best is None:
        raise ValueError("Cannot select a threshold from empty scores")
    return best[1], best[2]


def nested_classical_selection(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Iterable[dict[str, object]],
    make_estimator: Callable[[dict[str, object]], object],
    seed: int,
    n_splits: int = 5,
) -> tuple[dict[str, object], float, list[dict[str, object]]]:
    """Choose a candidate and threshold from inner out-of-fold predictions."""
    labels = np.asarray(y, dtype=np.int8)
    grouping = np.asarray(groups)
    available = min(n_splits, *(np.unique(grouping[labels == cls]).size for cls in (0, 1)))
    if available < 2:
        raise ValueError("Inner selection needs two independent groups per class")
    splitter = StratifiedGroupKFold(n_splits=available, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    selected: tuple[tuple[float, float, float, float], dict[str, object], float] | None = None
    for candidate in candidates:
        scores = np.empty(len(labels), dtype=float)
        for train, valid in splitter.split(np.zeros(len(labels)), labels, grouping):
            estimator = make_estimator(candidate)
            estimator.fit(X[train], labels[train])
            if hasattr(estimator, "decision_function"):
                scores[valid] = estimator.decision_function(X[valid])
            else:
                scores[valid] = estimator.predict_proba(X[valid])[:, 1]
        threshold, metrics = select_threshold(labels, scores)
        rank = tuple(float(metrics.get(name) or -1.0) for name in ("f1", "recall", "kappa", "balanced_accuracy"))
        rows.append({"candidate": candidate, "threshold": threshold, **metrics})
        if selected is None or rank > selected[0]:
            selected = (rank, candidate, threshold)
    if selected is None:
        raise ValueError("No inner candidate was evaluated")
    return selected[1], selected[2], rows
