#!/usr/bin/env python3
"""Verify local EEG labels against the public DataAlign.py semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts: Counter[int] = Counter()
    subject_lengths = {}
    for files in discover_subjects(args.raw_root, include_psg=True):
        alignment = build_alignment(
            files,
            match_official_psg_overlap=True,
            official_trim_last_second=True,
        )
        counts.update(int(value) for value in alignment.labels)
        subject_lengths[files.subject] = alignment.duration_sec
    observed = {label: counts[label] for label in EXPECTED_LITERAL_COUNTS}
    differences = {
        label: observed[label] - EXPECTED_LITERAL_COUNTS[label]
        for label in EXPECTED_LITERAL_COUNTS
    }
    result = {
        "status": "MATCH" if all(value == 0 for value in differences.values()) else "MISMATCH",
        "observed_counts": observed,
        "expected_counts_from_independent_audit": EXPECTED_LITERAL_COUNTS,
        "differences": differences,
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

