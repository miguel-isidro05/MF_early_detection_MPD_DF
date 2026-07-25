import numpy as np
import pandas as pd

from mpd_df.statistics import holm_adjust, paired_model_comparisons


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_paired_comparisons_use_shared_subjects_only() -> None:
    frame = pd.DataFrame(
        {
            "subject": ["01", "02", "03", "01", "02", "04"],
            "model": ["a", "a", "a", "b", "b", "b"],
            "f1": [0.6, 0.7, 0.8, 0.5, 0.6, 0.9],
        }
    )
    result = paired_model_comparisons(frame)
    assert result.loc[0, "n_pairs"] == 2
    assert result.loc[0, "alternative"] == "two-sided"

