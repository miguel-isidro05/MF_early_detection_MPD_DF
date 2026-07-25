#!/usr/bin/env python3
"""Verify local EEG labels against the public DataAlign.py semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from mpd_df.dataset import build_alignment, discover_subjects

EXPECTED_LITERAL_COUNTS = {
    0: 278545,
    1: 64001,
    2: 19973,
    3: 1498,
    4: 310,
    8: 593,
    9: 7484,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional full raw tree used to verify an EEG-only portable subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts: Counter[int] = Counter()
    subject_lengths = {}
    target_subjects = discover_subjects(args.raw_root, include_psg=True)
    reference = (
        {
            files.subject: build_alignment(files)
            for files in discover_subjects(args.reference_root, include_psg=True)
        }
        if args.reference_root
        else {}
    )
    reference_differences = []
    for files in target_subjects:
        alignment = build_alignment(
            files,
            match_official_psg_overlap=True,
            official_trim_last_second=True,
        )
        counts.update(int(value) for value in alignment.labels)
        subject_lengths[files.subject] = alignment.duration_sec
        if reference:
            expected = reference.get(files.subject)
            if expected is None:
                reference_differences.append(f"{files.subject}: missing from reference")
            elif (
                alignment.start_clock_sec != expected.start_clock_sec
                or alignment.eeg_offset_sec != expected.eeg_offset_sec
                or not np.array_equal(alignment.labels, expected.labels)
            ):
                reference_differences.append(f"{files.subject}: alignment differs")
    observed = {label: counts[label] for label in EXPECTED_LITERAL_COUNTS}
    differences = {
        label: observed[label] - EXPECTED_LITERAL_COUNTS[label]
        for label in EXPECTED_LITERAL_COUNTS
    }
    aggregate_comparable = len(target_subjects) == 50
    status = (
        "MATCH"
        if (
            not reference_differences
            and (
                bool(reference)
                or (aggregate_comparable and all(value == 0 for value in differences.values()))
            )
        )
        else "MISMATCH"
    )
    result = {
        "status": status,
        "verification_mode": "reference_root" if reference else "aggregate_literal_counts",
        "observed_counts": observed,
        "expected_counts_from_independent_audit": (
            EXPECTED_LITERAL_COUNTS if aggregate_comparable else None
        ),
        "differences": differences if aggregate_comparable else None,
        "reference_differences": reference_differences,
        "total_seconds": sum(observed.values()),
        "subject_lengths": subject_lengths,
        "interpretation": (
            "This verifies label expansion and EEG/PSG overlap timing only. "
            "It does not reproduce MSCNN-CAM training or Table 9."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "MATCH":
        raise SystemExit("Local alignment differs from the independent literal audit")


if __name__ == "__main__":
    main()
