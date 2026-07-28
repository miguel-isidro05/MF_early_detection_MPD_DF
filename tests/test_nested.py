import numpy as np

from mpd_df.nested import best_epoch_count, select_threshold


def test_select_threshold_prefers_fatigue_recall_when_f1_ties() -> None:
    threshold, metrics = select_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert 0.2 < threshold <= 0.8
    assert metrics["f1"] == 1.0


def test_best_epoch_count_uses_minimum_validation_loss() -> None:
    history = [
        {"epoch": 0, "validation_loss": 0.8},
        {"epoch": 1, "validation_loss": 0.4},
        {"epoch": 2, "validation_loss": 0.6},
    ]
    assert best_epoch_count(history) == 2


def test_best_epoch_count_rejects_missing_validation() -> None:
    history = [{"epoch": 0, "validation_loss": np.nan}]
    try:
        best_epoch_count(history)
    except ValueError as error:
        assert "finite validation loss" in str(error)
    else:
        raise AssertionError("Expected missing validation loss to fail")
