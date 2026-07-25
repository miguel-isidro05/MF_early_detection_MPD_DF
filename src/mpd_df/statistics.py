"""Planned participant-level statistical comparisons."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_model_comparisons(
    subject_metrics: pd.DataFrame,
    metric: str = "f1",
    model_column: str = "model",
    subject_column: str = "subject",
) -> pd.DataFrame:
    """Two-sided Wilcoxon tests on matched participant-level metrics."""

    required = {model_column, subject_column, metric}
    missing = required - set(subject_metrics)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = []
    models = sorted(subject_metrics[model_column].unique())
    for first, second in combinations(models, 2):
        left = subject_metrics.loc[
            subject_metrics[model_column] == first,
            [subject_column, metric],
        ].rename(columns={metric: "first_value"})
        right = subject_metrics.loc[
            subject_metrics[model_column] == second,
            [subject_column, metric],
        ].rename(columns={metric: "second_value"})
        paired = left.merge(right, on=subject_column, how="inner").dropna()
        if len(paired) < 2:
            statistic, p_value = np.nan, np.nan
        elif np.allclose(paired["first_value"], paired["second_value"]):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                paired["first_value"],
                paired["second_value"],
                alternative="two-sided",
            )
        rows.append(
            {
                "model_a": first,
                "model_b": second,
                "metric": metric,
                "n_pairs": len(paired),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "alternative": "two-sided",
            }
        )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    result["p_holm"] = np.nan
    result.loc[valid, "p_holm"] = holm_adjust(result.loc[valid, "p_value"].to_numpy())
    return result

