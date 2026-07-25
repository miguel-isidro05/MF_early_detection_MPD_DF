"""Leakage-aware within-subject and subject-independent splits."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .constants import GLOBAL_SEED


def validate_disjoint_groups(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    group_column: str = "group_id",
) -> None:
    train_groups = set(metadata.iloc[train_indices][group_column])
    test_groups = set(metadata.iloc[test_indices][group_column])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Group leakage detected: {sorted(overlap)[:5]}")


def within_subject_splits(
    metadata: pd.DataFrame,
    y: np.ndarray,
    requested_splits: int = 5,
    seed: int = GLOBAL_SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    subjects = metadata["subject"].astype(str).unique()
    if len(subjects) != 1:
        raise ValueError("within_subject_splits requires exactly one subject")
    groups = metadata["group_id"].to_numpy()
    class_group_counts = [
        np.unique(groups[y == class_id]).size for class_id in np.unique(y)
    ]
    n_splits = min(requested_splits, *class_group_counts)
    if n_splits < 2:
        raise ValueError("At least two independent groups per class are required")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        validate_disjoint_groups(metadata, train, test)
        yield train, test


def loso_splits(metadata: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = metadata["subject"].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(np.zeros(len(metadata)), groups=groups):
        train_subjects = set(groups[train])
        test_subjects = set(groups[test])
        if train_subjects & test_subjects:
            raise AssertionError("Subject leakage detected")
        yield train, test

