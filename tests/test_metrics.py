import numpy as np

from mpd_df.metrics import binary_metrics


def test_binary_metrics_use_fatigue_as_positive_class() -> None:
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_binary_metrics_mark_single_class_kappa_as_undefined() -> None:
    metrics = binary_metrics(np.array([0, 0]), np.array([0, 0]))
    assert metrics["kappa"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["n_negative"] == 2
    assert metrics["n_positive"] == 0
