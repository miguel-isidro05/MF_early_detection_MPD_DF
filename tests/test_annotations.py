from pathlib import Path

import numpy as np
import pytest

from mpd_df.annotations import (
    AnnotationChange,
    expand_change_points,
    map_binary_task,
    parse_annotation_line,
    read_annotation_file,
)


def test_parse_numeric_and_text_labels() -> None:
    numeric = parse_annotation_line("13:53:18,1,0", 1)
    artifact = parse_annotation_line("14:00:18,15,Severe Artifacts", 2)
    assert numeric.label == 0
    assert numeric.block_index == 1
    assert artifact.label == 9


def test_read_accepts_timestamp_block_disagreement(tmp_path: Path) -> None:
    path = tmp_path / "jitter.txt"
    path.write_text("13:00:00,1,0\n13:01:00,2,1\n")
    assert len(read_annotation_file(path)) == 2


def test_expand_preserves_30_second_block_ids_after_crop() -> None:
    changes = [
        AnnotationChange(100, 1, 0, 1),
        AnnotationChange(190, 4, 1, 2),
    ]
    labels, blocks = expand_change_points(changes, start_clock_sec=130, duration_sec=120)
    assert np.all(labels[:60] == 0)
    assert np.all(labels[60:] == 1)
    assert blocks.tolist() == [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30


@pytest.mark.parametrize(
    ("task", "expected_keep", "expected_y"),
    [
        ("A", [True, True, False, False, False, False], [0, 1]),
        ("B", [True, True, True, False, False, False], [0, 1, 1]),
        ("C", [True, True, True, True, False, False], [0, 1, 1, 1]),
    ],
)
def test_task_mapping(task: str, expected_keep: list[bool], expected_y: list[int]) -> None:
    labels = np.array([0, 1, 2, 4, 8, 9])
    keep, y = map_binary_task(labels, task)
    assert keep.tolist() == expected_keep
    assert y.tolist() == expected_y
