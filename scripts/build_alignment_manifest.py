#!/usr/bin/env python3
"""Persist the PSG-derived timing needed by an EEG-only portable dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mpd_df.dataset import build_alignment, discover_subjects, read_edf_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for files in discover_subjects(args.raw_root, include_psg=True):
        if files.psg is None:
            raise ValueError(f"PSG file is required while building subject {files.subject}")
        eeg = read_edf_header(files.eeg)
        psg = read_edf_header(files.psg)
        alignment = build_alignment(files)
        rows.append(
            {
                "subject": files.subject,
                "eeg_start_clock_sec": int(eeg["start_clock_sec"]),
                "eeg_duration_sec": float(eeg["duration_sec"]),
                "psg_start_clock_sec": int(psg["start_clock_sec"]),
                "psg_duration_sec": float(psg["duration_sec"]),
                "aligned_start_clock_sec": alignment.start_clock_sec,
                "eeg_offset_sec": alignment.eeg_offset_sec,
                "duration_sec": alignment.duration_sec,
                "official_trim_last_second": True,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} subjects to {args.output}")


if __name__ == "__main__":
    main()
