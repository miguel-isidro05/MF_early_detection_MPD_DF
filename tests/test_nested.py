import numpy as np

from mpd_df.nested import select_threshold


def test_select_threshold_prefers_fatigue_recall_when_f1_ties() -> None:
    threshold, metrics = select_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert 0.2 < threshold <= 0.8
    assert metrics["f1"] == 1.0
