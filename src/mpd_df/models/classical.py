"""Classical PSD baselines."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..constants import GLOBAL_SEED


def make_psd_svm(
    seed: int = GLOBAL_SEED,
    c: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=c,
                    class_weight=class_weight,
                    random_state=seed,
                    dual="auto",
                ),
            ),
        ]
    )


def make_random_forest(
    seed: int = GLOBAL_SEED,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    class_weight: str | dict[int, float] | None = "balanced_subsample",
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
