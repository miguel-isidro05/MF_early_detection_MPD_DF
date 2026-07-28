import pandas as pd

from scripts.assemble_final_task_a_results import context_aggregation_metrics


def test_context_aggregation_uses_clock_bins_not_true_label_boundaries() -> None:
    predictions = pd.DataFrame(
        {
            "protocol": ["within_subject"] * 4,
            "model": ["eegnet"] * 4,
            "subject": ["01"] * 4,
            "group_id": ["01:0"] * 4,
            "window_start_sec": [0, 1, 2, 3],
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 0, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9],
        }
    )

    _, source = context_aggregation_metrics(predictions, contexts=(4,))

    assert len(source) == 1
    assert source.iloc[0]["target_fraction"] == 0.5
