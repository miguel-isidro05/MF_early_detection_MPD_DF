"""Parsing and temporal expansion of MPD-DF physician annotations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import LABEL_NAMES, TASKS

TEXT_LABELS = {
    "Signal Abnormality": 8,
    "Severe Artifacts": 9,
}


@dataclass(frozen=True)
class AnnotationChange:
    """One change point from an MPD-DF annotation file."""

    timestamp_sec: int
    block_index: int
    label: int
    line_number: int


def clock_to_seconds(value: str) -> int:
    parsed = datetime.strptime(value.strip(), "%H:%M:%S")
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def parse_annotation_line(line: str, line_number: int) -> AnnotationChange:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Line {line_number}: expected 3 comma-separated fields")
    label_text = parts[2]
    if label_text in TEXT_LABELS:
        label = TEXT_LABELS[label_text]
    else:
        try:
            label = int(label_text)
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: unknown label {label_text!r}") from exc
    if label not in LABEL_NAMES:
        raise ValueError(f"Line {line_number}: unsupported label {label}")
    return AnnotationChange(
        timestamp_sec=clock_to_seconds(parts[0]),
        block_index=int(parts[1]),
        label=label,
        line_number=line_number,
    )


def read_annotation_file(path: str | Path) -> list[AnnotationChange]:
    changes = [
        parse_annotation_line(line.strip(), line_number)
        for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1)
        if line.strip()
    ]
    validate_annotation_changes(changes)
    return changes


def validate_annotation_changes(
    changes: Iterable[AnnotationChange],
) -> None:
    rows = list(changes)
    if not rows:
        raise ValueError("Annotation file is empty")
    for previous, current in zip(rows, rows[1:]):
        if current.timestamp_sec <= previous.timestamp_sec:
            raise ValueError(
                f"Line {current.line_number}: timestamps must be strictly increasing"
            )
        if current.block_index < previous.block_index:
            raise ValueError(
                f"Line {current.line_number}: block indices must be non-decreasing"
            )


def annotation_timing_diagnostics(
    changes: Iterable[AnnotationChange],
    block_duration_sec: int = 30,
) -> dict[str, int | float]:
    """Quantify disagreement between timestamps and the auxiliary block column."""

    rows = list(changes)
    validate_annotation_changes(rows)
    differences = []
    duplicate_indices = 0
    for previous, current in zip(rows, rows[1:]):
        elapsed = current.timestamp_sec - previous.timestamp_sec
        expected = (current.block_index - previous.block_index) * block_duration_sec
        differences.append(elapsed - expected)
        duplicate_indices += int(current.block_index == previous.block_index)
    absolute = [abs(value) for value in differences]
    return {
        "duplicate_consecutive_block_indices": duplicate_indices,
        "max_abs_timestamp_block_disagreement_sec": max(absolute, default=0),
        "mean_abs_timestamp_block_disagreement_sec": float(np.mean(absolute)) if absolute else 0.0,
    }


def expand_change_points(
    changes: Iterable[AnnotationChange],
    start_clock_sec: int,
    duration_sec: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand change points to second-level labels and original block ids.

    The first change point is effective at its timestamp. Seconds before it are
    excluded by alignment. The final label extends to the supplied signal end,
    matching the public repository's documented policy.
    """

    rows = list(changes)
    validate_annotation_changes(rows)
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if start_clock_sec < rows[0].timestamp_sec:
        raise ValueError("Aligned start cannot precede the first annotation")

    offsets = np.asarray([row.timestamp_sec - start_clock_sec for row in rows], dtype=int)
    labels = np.full(duration_sec, -1, dtype=np.int8)
    block_ids = np.full(duration_sec, -1, dtype=np.int32)

    for index, row in enumerate(rows):
        begin = max(0, int(offsets[index]))
        end = duration_sec if index == len(rows) - 1 else min(duration_sec, int(offsets[index + 1]))
        if begin >= duration_sec or end <= 0:
            continue
        seconds = np.arange(begin, end, dtype=np.int32)
        labels[begin:end] = row.label
        # The annotation's second column is only approximately related to 30 s
        # screens and can repeat. Fixed temporal bins are therefore used as the
        # conservative grouping unit; labels still follow official timestamps.
        block_ids[begin:end] = seconds // 30

    if np.any(labels < 0) or np.any(block_ids < 0):
        raise ValueError("Alignment produced unlabeled seconds")
    return labels, block_ids


def map_binary_task(labels: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a keep mask and binary labels for task A, B, or C."""

    try:
        definition = TASKS[task.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}") from exc
    negative = np.isin(labels, list(definition["negative"]))
    positive = np.isin(labels, list(definition["positive"]))
    keep = negative | positive
    binary = positive[keep].astype(np.int8)
    return keep, binary
