#!/usr/bin/env python3
"""Build resumable per-subject PSD feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mpd_df.annotations import map_binary_task
from mpd_df.constants import EDF_CHANNELS, PAPER_ANALYTICAL_CHANNELS
from mpd_df.dataset import (
    build_alignment,
    build_window_index,
    discover_subjects,
    iter_preprocessed_windows,
)
from mpd_df.features import bandpower_features
from mpd_df.preprocessing import (
    ANNOTATION_CLASSIFICATION,
    ANNOTATION_GLOBAL_ZSCORE,
    ANNOTATION_MICROVOLT,
    ANNOTATION_VISUALIZATION,
    MNE_EEGLAB_LIKE,
    PHYSIOLOGICAL_GLOBAL_ZSCORE,
    PHYSIOLOGICAL_MICROVOLT,
    PHYSIOLOGICAL_VALIDATION,
    PSD_CLASSIFICATION,
    REFERENCE_UNSPECIFIED,
)

PREPROCESSING = {
    config.name: config
    for config in (
        REFERENCE_UNSPECIFIED,
        PHYSIOLOGICAL_VALIDATION,
        PHYSIOLOGICAL_GLOBAL_ZSCORE,
        PHYSIOLOGICAL_MICROVOLT,
        PSD_CLASSIFICATION,
        ANNOTATION_CLASSIFICATION,
        ANNOTATION_GLOBAL_ZSCORE,
        ANNOTATION_MICROVOLT,
        ANNOTATION_VISUALIZATION,
        MNE_EEGLAB_LIKE,
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
    parser.add_argument(
        "--filter-scope",
        choices=("group_bounded", "subject_continuous"),
        default="group_bounded",
    )
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--transition-guard-sec", type=int, default=0)
    return parser.parse_args()


def file_signature(path: Path, raw_root: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256()
    edge_bytes = 1024 * 1024
    with path.open("rb") as stream:
        digest.update(stream.read(edge_bytes))
        if stat.st_size > edge_bytes:
            stream.seek(max(0, stat.st_size - edge_bytes))
            digest.update(stream.read(edge_bytes))
    return {
        "path": (
            path.name
            if path.parent.absolute() == raw_root.absolute()
            else f"{path.parent.name}/{path.name}"
        ),
        "size": stat.st_size,
        "edge_sha256": digest.hexdigest(),
    }


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


def main() -> None:
    args = parse_args()
    if args.filter_context_sec < 0:
        raise ValueError("filter_context_sec must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PREPROCESSING[args.preprocessing]
    picks = EDF_CHANNELS if args.montage == "edf32" else PAPER_ANALYTICAL_CHANNELS
    requested = {value.zfill(2) for value in args.subjects or []}
    subjects = [
        files
        for files in discover_subjects(args.raw_root, include_psg=True)
        if not requested or files.subject in requested
    ]
    if requested and requested != {files.subject for files in subjects}:
        raise ValueError("One or more requested subjects were not found")
    manifest = {
        "task": args.task,
        "preprocessing": config.to_dict(),
        "montage": args.montage,
        "channels": list(picks),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "filter_context_sec": args.filter_context_sec,
        "filter_scope": args.filter_scope,
        "transition_guard_sec": args.transition_guard_sec,
        "subjects_requested": sorted(requested),
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
            "filter_scope": args.filter_scope,
            "transition_guard_sec": args.transition_guard_sec,
            "subjects_requested": sorted(requested),
            "code_hash": manifest["code_hash"],
        }
    )
    manifest["config_fingerprint"] = config_fingerprint
    for subject_index, files in enumerate(subjects, start=1):
        print(
            f"[PSD cache] subject {files.subject} "
            f"({subject_index}/{len(subjects)})",
            flush=True,
        )
        output = args.output_dir / f"subject_{files.subject}.npz"
        if files.psg is None and files.alignment_manifest is None:
            raise ValueError(
                f"Missing PSG timing file and alignment manifest for subject {files.subject}"
            )
        subject_source = {
            "eeg": file_signature(files.eeg, args.raw_root),
            "annotation": file_signature(files.annotation, args.raw_root),
            "psg": (
                file_signature(files.psg, args.raw_root)
                if files.psg is not None
                else file_signature(files.alignment_manifest, args.raw_root)
            ),
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
        if args.transition_guard_sec < 0:
            raise ValueError("transition_guard_sec must be non-negative")
        if args.transition_guard_sec:
            index = index.loc[
                index["distance_to_transition_sec"] >= args.transition_guard_sec
            ].reset_index(drop=True)
        keep, y = map_binary_task(index["label"].to_numpy(), args.task)
        index = index.loc[keep].reset_index(drop=True)
        feature_batches = []
        feature_names = None
        for windows, _, output_sfreq in iter_preprocessed_windows(
            files,
            alignment,
            index,
            picks,
            config,
            args.batch_size,
            args.filter_context_sec,
            args.filter_scope,
        ):
            features, names = bandpower_features(windows, output_sfreq, picks)
            if config.normalization == "per_window_channel_zscore":
                names = [
                    name.replace("_abs_", "_zscore_psd_")
                    for name in names
                ]
            feature_batches.append(features)
            feature_names = names
        X = np.concatenate(feature_batches) if feature_batches else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(
            output,
            X=X,
            y=y,
            label=index["label"].to_numpy(dtype=np.int8),
            subject=index["subject"].astype(str).to_numpy(dtype="U"),
            window_start_sec=index["window_start_sec"].to_numpy(),
            block_id=index["block_id"].to_numpy(),
            group_id=index["group_id"].astype(str).to_numpy(dtype="U"),
            distance_to_transition_sec=index[
                "distance_to_transition_sec"
            ].to_numpy(dtype=np.float32),
            feature_names=np.asarray(feature_names or [], dtype="U"),
            cache_fingerprint=np.asarray(cache_fingerprint),
            config_fingerprint=np.asarray(config_fingerprint),
        )
        manifest["subjects"].append(files.subject)
        manifest["subject_fingerprints"][files.subject] = cache_fingerprint
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PSD cache] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
