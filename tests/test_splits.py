import numpy as np
import pandas as pd

from mpd_df.splits import loso_splits, within_subject_splits


def test_within_subject_keeps_blocks_disjoint() -> None:
    groups = np.repeat([f"01:{index}" for index in range(10)], 3)
    y = np.repeat([0, 1] * 5, 3)
    metadata = pd.DataFrame({"subject": "01", "group_id": groups})
    splits = list(within_subject_splits(metadata, y, requested_splits=5))
    assert len(splits) == 5
    for train, test in splits:
        assert set(groups[train]).isdisjoint(groups[test])


def test_loso_keeps_subjects_disjoint() -> None:
    subjects = np.repeat(["01", "02", "03"], 4)
    metadata = pd.DataFrame(
        {
            "subject": subjects,
            "group_id": [f"{subject}:{index}" for subject in subjects for index in [0]][:12],
        }
    )
    splits = list(loso_splits(metadata))
    assert len(splits) == 3
    for train, test in splits:
        assert set(subjects[train]).isdisjoint(subjects[test])

