"""Dataset discovery, EDF metadata, alignment, and lazy EEG windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterator, Sequence

import mne
import numpy as np
import pandas as pd

from .annotations import expand_change_points, read_annotation_file

SUBJECT_PATTERN = re.compile(r"MPDDF_raw_(\d{2})_EEG\.edf$")
EDF_ASSUMED_UNITS = "uV"
EDF_ASSUMED_SCALE_TO_VOLTS = 1e-6


@dataclass(frozen=True)
class SubjectFiles:
    subject: str
    eeg: Path
    annotation: Path
    psg: Path | None = None


@dataclass(frozen=True)
class Alignment:
    subject: str
    start_clock_sec: int
    duration_sec: int
    eeg_offset_sec: int
    labels: np.ndarray
    block_ids: np.ndarray


def discover_subjects(raw_root: str | Path, include_psg: bool = True) -> list[SubjectFiles]:
    root = Path(raw_root)
    eeg_dir = root / "EEG"
    annotation_dir = root / "Annotation"
    psg_dir = root / "PSG"
    subjects: list[SubjectFiles] = []
    for eeg_path in sorted(eeg_dir.glob("MPDDF_raw_*_EEG.edf")):
        match = SUBJECT_PATTERN.search(eeg_path.name)
        if not match:
            continue
        subject = match.group(1)
        annotation = annotation_dir / f"MPDDF_raw_{subject}_Annotation.txt"
        psg = psg_dir / f"MPDDF_raw_{subject}_PSG.edf"
        if not annotation.exists():
            raise FileNotFoundError(annotation)
        subjects.append(
            SubjectFiles(
                subject=subject,
                eeg=eeg_path,
                annotation=annotation,
                psg=psg if include_psg and psg.exists() else None,
            )
        )
    if not subjects:
        raise FileNotFoundError(f"No MPD-DF EEG files found under {eeg_dir}")
    return subjects


def _clock_seconds(meas_date: datetime | None) -> int:
    if meas_date is None:
        raise ValueError("EDF is missing meas_date; clock alignment is impossible")
    return meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second


def read_edf_header(path: str | Path) -> dict[str, object]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    return {
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": int(len(raw.ch_names)),
        "ch_names": tuple(raw.ch_names),
        "n_samples": int(raw.n_times),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "start_clock_sec": _clock_seconds(raw.info["meas_date"]),
    }


def build_alignment(
    files: SubjectFiles,
    match_official_psg_overlap: bool = True,
    official_trim_last_second: bool = True,
) -> Alignment:
    eeg = read_edf_header(files.eeg)
    changes = read_annotation_file(files.annotation)
    annotation_start = changes[0].timestamp_sec
    start = max(int(eeg["start_clock_sec"]), annotation_start)
    end = int(eeg["start_clock_sec"] + float(eeg["duration_sec"]))

    if match_official_psg_overlap:
        if files.psg is None:
            raise ValueError("PSG header is required to match official alignment")
        psg = read_edf_header(files.psg)
        start = max(start, int(psg["start_clock_sec"]))
        end = min(end, int(psg["start_clock_sec"] + float(psg["duration_sec"])))

    if end < changes[-1].timestamp_sec:
        raise ValueError(
            f"Subject {files.subject}: aligned end {end} precedes final annotation "
            f"{changes[-1].timestamp_sec}"
        )
    duration = end - start - int(official_trim_last_second)
    if duration <= 0:
        raise ValueError("No positive-duration overlap after alignment")
    labels, block_ids = expand_change_points(changes, start, duration)
    return Alignment(
        subject=files.subject,
        start_clock_sec=start,
        duration_sec=duration,
        eeg_offset_sec=start - int(eeg["start_clock_sec"]),
        labels=labels,
        block_ids=block_ids,
    )


def build_window_index(
    files: SubjectFiles,
    alignment: Alignment,
    window_sec: int = 1,
    stride_sec: int | None = None,
) -> pd.DataFrame:
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    stride_sec = window_sec if stride_sec is None else stride_sec
    if stride_sec <= 0:
        raise ValueError("stride_sec must be positive")
    rows: list[dict[str, int | str]] = []
    for start in range(0, alignment.duration_sec - window_sec + 1, stride_sec):
        end = start + window_sec
        window_labels = alignment.labels[start:end]
        window_blocks = alignment.block_ids[start:end]
        if np.unique(window_labels).size != 1 or np.unique(window_blocks).size != 1:
            continue
        rows.append(
            {
                "subject": files.subject,
                "window_start_sec": start,
                "eeg_start_sec": alignment.eeg_offset_sec + start,
                "window_sec": window_sec,
                "label": int(window_labels[0]),
                "block_id": int(window_blocks[0]),
                "group_id": f"{files.subject}:{int(window_blocks[0])}",
            }
        )
    return pd.DataFrame(rows)


def iter_eeg_windows(
    files: SubjectFiles,
    index: pd.DataFrame,
    picks: Sequence[str] | None = None,
    batch_size: int = 256,
) -> Iterator[tuple[np.ndarray, pd.DataFrame]]:
    """Yield lazy batches shaped (windows, channels, samples)."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
    if picks is not None:
        missing = sorted(set(picks) - set(raw.ch_names))
        if missing:
            raise ValueError(f"Missing requested channels: {missing}")
        raw.pick(list(picks))
    sfreq = int(round(raw.info["sfreq"]))
    for begin in range(0, len(index), batch_size):
        metadata = index.iloc[begin : begin + batch_size].copy()
        windows = []
        for row in metadata.itertuples(index=False):
            first = int(round(row.eeg_start_sec * sfreq))
            last = first + int(round(row.window_sec * sfreq))
            windows.append(
                raw.get_data(start=first, stop=last) * EDF_ASSUMED_SCALE_TO_VOLTS
            )
        yield np.stack(windows).astype(np.float32, copy=False), metadata
