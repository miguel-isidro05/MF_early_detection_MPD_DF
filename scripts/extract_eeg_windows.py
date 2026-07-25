#!/usr/bin/env python3
"""Create memory-mapped EEG window caches for portable deep-model training."""

from __future__ import annotations

import argparse
import hashlib
from itertools import chain
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mpd_df.annotations import map_binary_task
from mpd_df.constants import EDF_CHANNELS, PAPER_ANALYTICAL_CHANNELS
from mpd_df.dataset import (
    build_alignment,
    build_window_index,
    discover_subjects,
    iter_preprocessed_windows,
)
from mpd_df.preprocessing import (
    ANNOTATION_VISUALIZATION,
    PHYSIOLOGICAL_VALIDATION,
    REFERENCE_UNSPECIFIED,
)

PREPROCESSING = {
    config.name: config
    for config in (
        REFERENCE_UNSPECIFIED,
        PHYSIOLOGICAL_VALIDATION,
        ANNOTATION_VISUALIZATION,
    )
}


def edge_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return {
        "logical_path": f"{path.parent.name}/{path.name}",
        "size": stat.st_size,
        "edge_sha256": digest.hexdigest(),
    }


def code_hash() -> str:
    project = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((project / "src" / "mpd_df").rglob("*.py")):
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("A", "B", "C"), required=True)
    parser.add_argument("--preprocessing", choices=sorted(PREPROCESSING), required=True)
    parser.add_argument("--montage", choices=("edf32", "paper28"), default="edf32")
    parser.add_argument("--window-sec", type=int, default=1)
    parser.add_argument("--stride-sec", type=int, default=1)
    parser.add_argument("--filter-scope", choices=("group_bounded", "subject_continuous"))
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--max-windows-per-class", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_balanced_subset(index, task: str, maximum: int | None):
    keep, y = map_binary_task(index["label"].to_numpy(), task)
    selected = index.loc[keep].copy()
    selected["target"] = y
    if maximum is None:
        return selected.reset_index(drop=True)
    parts = [
        rows.iloc[:maximum]
        for _, rows in selected.groupby("target", sort=True)
    ]
    return (
        selected.iloc[0:0]
        if not parts
        else pd.concat(parts).sort_index().reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    picks = EDF_CHANNELS if args.montage == "edf32" else PAPER_ANALYTICAL_CHANNELS
    config = PREPROCESSING[args.preprocessing]
    filter_scope = args.filter_scope or "group_bounded"
    requested = {value.zfill(2) for value in args.subjects or []}
    subjects = [
        files
        for files in discover_subjects(args.raw_root, include_psg=True)
        if not requested or files.subject in requested
    ]
    if requested != {files.subject for files in subjects}:
        raise ValueError("One or more requested subjects were not found")
    manifest = {
        "task": args.task,
        "preprocessing": config.to_dict(),
        "montage": args.montage,
        "channels": list(picks),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "filter_scope": filter_scope,
        "subjects": [],
        "code_hash": code_hash(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    manifest["config_fingerprint"] = fingerprint

    for files in subjects:
        data_path = args.output_dir / f"subject_{files.subject}_X.npy"
        metadata_path = args.output_dir / f"subject_{files.subject}_metadata.csv"
        if (data_path.exists() or metadata_path.exists()) and not args.overwrite:
            raise FileExistsError(f"Cache already exists for subject {files.subject}")
        alignment = build_alignment(files)
        index = build_window_index(
            files,
            alignment,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )
        index = choose_balanced_subset(index, args.task, args.max_windows_per_class)
        batches = iter_preprocessed_windows(
            files,
            alignment,
            index,
            picks,
            config,
            filter_scope=filter_scope,
        )
        try:
            first_windows, first_metadata, output_sfreq = next(batches)
        except StopIteration as exc:
            raise ValueError(f"No eligible windows for subject {files.subject}") from exc
        n_times = first_windows.shape[-1]
        mmap = np.lib.format.open_memmap(
            data_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(index), len(picks), n_times),
        )
        ordered_metadata = []
        cursor = 0
        for windows, metadata, _ in chain(
            [(first_windows, first_metadata, output_sfreq)],
            batches,
        ):
            count = len(windows)
            mmap[cursor : cursor + count] = windows
            ordered_metadata.append(metadata)
            cursor += count
        mmap.flush()
        metadata = pd.concat(ordered_metadata, ignore_index=True)
        _, targets = map_binary_task(metadata["label"].to_numpy(), args.task)
        metadata["target"] = targets
        metadata.to_csv(metadata_path, index=False)
        manifest["subjects"].append(
            {
                "subject": files.subject,
                "data": data_path.name,
                "metadata": metadata_path.name,
                "n_windows": len(metadata),
                "n_channels": len(picks),
                "n_times": n_times,
                "sfreq": output_sfreq,
                "source": {
                    "eeg": edge_signature(files.eeg),
                    "annotation": edge_signature(files.annotation),
                    "timing": edge_signature(
                        files.psg if files.psg is not None else files.alignment_manifest
                    ),
                },
            }
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
