#!/usr/bin/env python3
"""Audit nested grouped within-subject eligibility before model extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mpd_df.annotations import map_binary_task
from mpd_df.dataset import build_alignment, build_window_index, discover_subjects
from mpd_df.deep_training import build_deep_split_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        build_window_index(files, build_alignment(files))
        for files in discover_subjects(args.raw_root, include_psg=True)
    ]
    complete = pd.concat(frames, ignore_index=True)
    summary = []
    for task in ("A", "B", "C"):
        keep, y = map_binary_task(complete["label"].to_numpy(), task)
        metadata = complete.loc[keep].reset_index(drop=True)
        plans, exclusions = build_deep_split_plan(
            metadata,
            y,
            protocol="within_subject",
            seed=args.seed,
        )
        excluded = {str(row["subject"]).zfill(2) for row in exclusions}
        eligible = sorted(set(metadata["subject"].astype(str).str.zfill(2)) - excluded)
        (args.output_dir / f"eligible_within_task_{task.lower()}.txt").write_text(
            "\n".join(eligible) + "\n"
        )
        pd.DataFrame(exclusions, columns=["scope", "subject", "fold", "reason"]).to_csv(
            args.output_dir / f"excluded_within_task_{task.lower()}.csv",
            index=False,
        )
        summary.append(
            {
                "task": task,
                "total_subjects": metadata["subject"].nunique(),
                "eligible_subjects": len(eligible),
                "excluded_subjects": len(excluded),
                "valid_outer_folds": len(plans),
            }
        )
    pd.DataFrame(summary).to_csv(args.output_dir / "within_subject_eligibility.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
