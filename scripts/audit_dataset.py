#!/usr/bin/env python3
"""Generate the mandatory MPD-DF dataset audit artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from mpd_df.annotations import annotation_timing_diagnostics, read_annotation_file
from mpd_df.constants import EDF_CHANNELS, LABEL_NAMES, PAPER_ANALYTICAL_CHANNELS
from mpd_df.dataset import (
    EDF_ASSUMED_SCALE_TO_VOLTS,
    EDF_ASSUMED_UNITS,
    build_alignment,
    discover_subjects,
    read_edf_header,
)

PAPER_NORMALIZED_COUNTS = {0: 266718, 1: 60292, 2: 14789, 3: 760, 4: 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-qc", choices=("none", "sampled", "full"), default="sampled")
    parser.add_argument("--sample-seconds", type=int, default=60)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def signal_qc(
    path: Path,
    mode: str,
    sample_seconds: int,
) -> dict[str, float | int | None]:
    if mode == "none":
        return {
            "finite_fraction": None,
            "min_uv": None,
            "max_uv": None,
            "sampled_samples": 0,
        }
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    sfreq = int(round(raw.info["sfreq"]))
    chunk = sample_seconds * sfreq
    if mode == "full":
        starts = range(0, raw.n_times, chunk)
    else:
        starts = sorted(
            {
                0,
                max(0, raw.n_times // 2 - chunk // 2),
                max(0, raw.n_times - chunk),
            }
        )
    finite = 0
    total = 0
    minimum = np.inf
    maximum = -np.inf
    for start in starts:
        values = raw.get_data(start=start, stop=min(raw.n_times, start + chunk))
        finite_mask = np.isfinite(values)
        finite += int(finite_mask.sum())
        total += int(values.size)
        if np.any(finite_mask):
            finite_values = (
                values[finite_mask] * EDF_ASSUMED_SCALE_TO_VOLTS * 1e6
            )
            minimum = min(minimum, float(finite_values.min()))
            maximum = max(maximum, float(finite_values.max()))
    return {
        "finite_fraction": finite / total if total else None,
        "min_uv": minimum if np.isfinite(minimum) else None,
        "max_uv": maximum if np.isfinite(maximum) else None,
        "sampled_samples": total,
    }


def first_transition(labels: np.ndarray, source: int, target: int) -> int | None:
    locations = np.flatnonzero((labels[:-1] == source) & (labels[1:] == target))
    return int(locations[0] + 1) if locations.size else None


def normalized_118_minutes(labels: np.ndarray) -> np.ndarray:
    """Apply the descriptor's duration normalization as literally as possible."""

    start = min(60, len(labels))
    return labels[start : min(len(labels), start + 118 * 60)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = discover_subjects(args.raw_root, include_psg=True)
    dataset_rows = []
    subject_rows = []
    channel_rows = []
    label_rows = []
    artifact_rows = []
    all_counts: Counter[int] = Counter()
    all_normalized_counts: Counter[int] = Counter()
    errors: list[str] = []

    for files in subjects:
        try:
            header = read_edf_header(files.eeg)
            alignment = build_alignment(files, match_official_psg_overlap=True)
            changes = read_annotation_file(files.annotation)
            timing = annotation_timing_diagnostics(changes)
            qc = signal_qc(files.eeg, args.signal_qc, args.sample_seconds)
            counts = Counter(int(value) for value in alignment.labels)
            normalized_counts = Counter(int(value) for value in normalized_118_minutes(alignment.labels))
            all_counts.update(counts)
            all_normalized_counts.update(normalized_counts)
            unique_blocks = {
                label: np.unique(alignment.block_ids[alignment.labels == label]).size
                for label in LABEL_NAMES
            }
            transition_count = int(np.count_nonzero(np.diff(alignment.labels)))
            subject_rows.append(
                {
                    "subject": files.subject,
                    "sfreq": header["sfreq"],
                    "edf_declared_unit": "corrupted_microvolt_bytes_exposed_as_n/a",
                    "analysis_assumed_unit": EDF_ASSUMED_UNITS,
                    "n_channels": header["n_channels"],
                    "n_samples": header["n_samples"],
                    "raw_duration_sec": header["duration_sec"],
                    "aligned_duration_sec": alignment.duration_sec,
                    "alignment_offset_sec": alignment.eeg_offset_sec,
                    "annotation_change_points": len(changes),
                    "annotation_blocks": int(np.unique(alignment.block_ids).size),
                    "first_wake_to_fatigue1_sec": first_transition(alignment.labels, 0, 1),
                    "total_label_transitions": transition_count,
                    "signal_abnormality_sec": counts[8],
                    "severe_artifacts_sec": counts[9],
                    "recording_truncated_lt_7080_sec": alignment.duration_sec < 7080,
                    "signal_label_mismatch_sec": float(header["duration_sec"])
                    - alignment.eeg_offset_sec
                    - alignment.duration_sec,
                    **timing,
                    **qc,
                }
            )
            for channel_index, channel in enumerate(header["ch_names"]):
                channel_rows.append(
                    {
                        "subject": files.subject,
                        "channel_index": channel_index,
                        "channel": channel,
                        "in_paper_analytical_28": channel in PAPER_ANALYTICAL_CHANNELS,
                    }
                )
            for label, name in LABEL_NAMES.items():
                label_rows.append(
                    {
                        "subject": files.subject,
                        "label": label,
                        "label_name": name,
                        "seconds": counts[label],
                        "normalized_118min_seconds": normalized_counts[label],
                        "annotation_blocks": unique_blocks[label],
                    }
                )
            artifact_rows.append(
                {
                    "subject": files.subject,
                    "signal_abnormality_sec": counts[8],
                    "severe_artifacts_sec": counts[9],
                    "artifact_total_sec": counts[8] + counts[9],
                }
            )
            for kind, path in (
                ("EEG", files.eeg),
                ("Annotation", files.annotation),
                ("PSG", files.psg),
            ):
                dataset_rows.append(
                    {
                        "subject": files.subject,
                        "kind": kind,
                        "path": str(path),
                        "size_bytes": path.stat().st_size if path else None,
                        "sha256": sha256(path) if path and kind == "Annotation" else "",
                        "exists": bool(path and path.exists()),
                    }
                )
        except Exception as exc:
            errors.append(f"{files.subject}: {type(exc).__name__}: {exc}")

    pd.DataFrame(dataset_rows).to_csv(args.output_dir / "dataset_inventory.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(args.output_dir / "subject_inventory.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(args.output_dir / "channel_inventory.csv", index=False)
    pd.DataFrame(label_rows).to_csv(args.output_dir / "label_inventory.csv", index=False)
    pd.DataFrame(artifact_rows).to_csv(args.output_dir / "artifact_inventory.csv", index=False)

    integrity = [
        "# File Integrity Report",
        "",
        f"- Subjects discovered: {len(subjects)}",
        f"- Subjects audited successfully: {len(subject_rows)}",
        f"- Errors: {len(errors)}",
        f"- Signal QC mode: `{args.signal_qc}`",
        "",
        "## Errors",
        "",
        *(f"- {error}" for error in errors),
    ]
    (args.output_dir / "file_integrity_report.md").write_text("\n".join(integrity) + "\n")

    comparison = []
    for label in range(5):
        comparison.append(
            {
                "label": label,
                "local_aligned_seconds": all_counts[label],
                "local_normalized_118min_seconds": all_normalized_counts[label],
                "paper_normalized_seconds": PAPER_NORMALIZED_COUNTS[label],
                "normalized_difference": all_normalized_counts[label]
                - PAPER_NORMALIZED_COUNTS[label],
            }
        )
    pd.DataFrame(comparison).to_csv(
        args.output_dir / "paper_table7_comparison.csv",
        index=False,
    )
    audit = {
        "subjects": len(subjects),
        "successful": len(subject_rows),
        "errors": errors,
        "unique_eeg_channel_count": sorted({row["n_channels"] for row in subject_rows}),
        "unique_sampling_rates": sorted({row["sfreq"] for row in subject_rows}),
        "edf_channel_order_matches_verified_32": all(
            tuple(row["channel"] for row in channel_rows if row["subject"] == subject.subject)
            == EDF_CHANNELS
            for subject in subjects
        ),
        "paper_comparison_note": (
            "Both complete official-overlap counts and a literal 118-minute normalization are "
            "reported. The remaining differences from Table 7 are unresolved and are not corrected."
        ),
        "class_comparison": comparison,
    }
    (args.output_dir / "dataset_audit.md").write_text(
        "# MPD-DF Dataset Audit\n\n```json\n"
        + json.dumps(audit, indent=2)
        + "\n```\n"
    )
    if errors:
        raise SystemExit(f"Audit completed with {len(errors)} subject errors")


if __name__ == "__main__":
    main()
