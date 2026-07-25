from pathlib import Path

import numpy as np

from mpd_df.annotations import AnnotationChange
from mpd_df.dataset import Alignment, SubjectFiles, build_window_index


def test_window_index_never_crosses_annotation_blocks() -> None:
    files = SubjectFiles("01", Path("dummy.edf"), Path("dummy.txt"))
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int8)
    blocks = np.array([1] * 30 + [2] * 30, dtype=np.int32)
    alignment = Alignment("01", 100, 60, 0, labels, blocks)
    index = build_window_index(files, alignment, window_sec=10, stride_sec=5)
    assert len(index) == 10
    assert not any(
        row.window_start_sec < 30 < row.window_start_sec + row.window_sec
        for row in index.itertuples()
    )
    assert set(index["group_id"]) == {"01:1", "01:2"}

