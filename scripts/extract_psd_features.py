#!/usr/bin/env python3
"""Build resumable per-subject PSD feature caches."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import mne

from mpd_df.annotations import map_binary_task
from mpd_df.constants import EDF_CHANNELS, PAPER_ANALYTICAL_CHANNELS
from mpd_df.dataset import (
    EDF_ASSUMED_SCALE_TO_VOLTS,
    EDF_ASSUMED_UNITS,
    build_alignment,
    build_window_index,
    discover_subjects,
)
from mpd_df.features import bandpower_features
from mpd_df.preprocessing import (
    ANNOTATION_VISUALIZATION,
    PHYSIOLOGICAL_VALIDATION,
    REFERENCE_UNSPECIFIED,
    preprocess_batch,
)

PREPROCESSING = {
    config.name: config
    for config in (
        REFERENCE_UNSPECIFIED,
        PHYSIOLOGICAL_VALIDATION,
        ANNOTATION_VISUALIZATION,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("A", "B", "C"), required=True)
    parser.add_argument("--preprocessing", choices=sorted(PREPROCESSING), required=True)
    parser.add_argument("--montage", choices=("edf32", "paper28"), default="edf32")
    parser.add_argument("--window-sec", type=int, default=1)
    parser.add_argument("--stride-sec", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--filter-context-sec", type=int, default=10)
    return parser.parse_args()


def file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def code_hash() -> str:
    source_root = Path(__file__).resolve().parents[1] / "src" / "mpd_df"
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def normalize_windows(windows: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "none":
        return windows
    if normalization != "per_window_channel_zscore":
        raise ValueError(f"Unsupported deferred normalization: {normalization}")
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True)
    return (windows - mean) / np.maximum(std, np.finfo(windows.dtype).eps)


def iter_context_filtered_windows(
    files,
    index,
    picks,
    config,
    batch_size: int,
    context_sec: int,
):
    """Filter contiguous signal chunks, then cut windows from their centers."""

    raw = mne.io.read_raw_edf(files.eeg, preload=False, verbose="ERROR")
    raw.pick(list(picks))
    source_sfreq = float(raw.info["sfreq"])
    filtering_config = replace(config, normalization="none")
    for begin in range(0, len(index), batch_size):
        metadata = index.iloc[begin : begin + batch_size].copy()
        first_sec = float(metadata["eeg_start_sec"].min())
        final_sec = float((metadata["eeg_start_sec"] + metadata["window_sec"]).max())
        read_start_sec = max(0.0, first_sec - context_sec)
        read_end_sec = min(raw.n_times / source_sfreq, final_sec + context_sec)
        first_sample = int(round(read_start_sec * source_sfreq))
        final_sample = int(round(read_end_sec * source_sfreq))
        continuous = (
            raw.get_data(start=first_sample, stop=final_sample)
            * EDF_ASSUMED_SCALE_TO_VOLTS
        )[None, ...]
        filtered, output_sfreq = preprocess_batch(
            continuous,
            source_sfreq,
            filtering_config,
        )
        filtered = filtered[0]
        windows = []
        for row in metadata.itertuples(index=False):
            local_start_sec = row.eeg_start_sec - read_start_sec
            local_start = int(round(local_start_sec * output_sfreq))
            local_end = local_start + int(round(row.window_sec * output_sfreq))
            windows.append(filtered[:, local_start:local_end])
        window_array = np.stack(windows).astype(np.float32, copy=False)
        yield normalize_windows(window_array, config.normalization), metadata, output_sfreq


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PREPROCESSING[args.preprocessing]
    picks = EDF_CHANNELS if args.montage == "edf32" else PAPER_ANALYTICAL_CHANNELS
    manifest = {
        "task": args.task,
        "preprocessing": config.to_dict(),
        "montage": args.montage,
        "channels": list(picks),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "filter_context_sec": args.filter_context_sec,
        "code_hash": code_hash(),
        "subjects": [],
        "subject_fingerprints": {},
    }
    config_fingerprint = stable_hash(
        {
            "task": args.task,
            "preprocessing": config.to_dict(),
            "montage": args.montage,
            "channels": list(picks),
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "filter_context_sec": args.filter_context_sec,
            "code_hash": manifest["code_hash"],
        }
    )
    manifest["config_fingerprint"] = config_fingerprint
    for files in discover_subjects(args.raw_root, include_psg=True):
        output = args.output_dir / f"subject_{files.subject}.npz"
        if files.psg is None:
            raise ValueError(f"Missing PSG timing file for subject {files.subject}")
        subject_source = {
            "eeg": file_signature(files.eeg),
            "annotation": file_signature(files.annotation),
            "psg": file_signature(files.psg),
        }
        cache_fingerprint = stable_hash(
            {"config_fingerprint": config_fingerprint, "source": subject_source}
        )
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as cached:
                observed = str(cached["cache_fingerprint"].item())
            if observed != cache_fingerprint:
                raise ValueError(
                    f"Cache fingerprint mismatch for subject {files.subject}; "
                    "use a new output directory or --overwrite"
                )
            manifest["subjects"].append(files.subject)
            manifest["subject_fingerprints"][files.subject] = cache_fingerprint
            continue
        alignment = build_alignment(files, match_official_psg_overlap=True)
        index = build_window_index(
            files,
            alignment,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )
        keep, y = map_binary_task(index["label"].to_numpy(), args.task)
        index = index.loc[keep].reset_index(drop=True)
        feature_batches = []
        feature_names = None
        for windows, _, output_sfreq in iter_context_filtered_windows(
            files,
            index,
            picks,
            config,
            args.batch_size,
            args.filter_context_sec,
        ):
            features, names = bandpower_features(windows, output_sfreq, picks)
            feature_batches.append(features)
            feature_names = names
        X = np.concatenate(feature_batches) if feature_batches else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(
            output,
            X=X,
            y=y,
            subject=index["subject"].astype(str).to_numpy(dtype="U"),
            window_start_sec=index["window_start_sec"].to_numpy(),
            block_id=index["block_id"].to_numpy(),
            group_id=index["group_id"].astype(str).to_numpy(dtype="U"),
            feature_names=np.asarray(feature_names or [], dtype="U"),
            cache_fingerprint=np.asarray(cache_fingerprint),
            config_fingerprint=np.asarray(config_fingerprint),
        )
        manifest["subjects"].append(files.subject)
        manifest["subject_fingerprints"][files.subject] = cache_fingerprint
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
