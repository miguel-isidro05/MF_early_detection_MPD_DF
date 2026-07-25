#!/usr/bin/env python3
"""Run grouped within-subject or LOSO PSD baselines from feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

from mpd_df.constants import GLOBAL_SEED
from mpd_df.experiment import (
    ExperimentContext,
    evaluate_classical_splits,
    finalize_experiment,
    initialize_experiment,
    save_experiment_results,
)
from mpd_df.metrics import binary_metrics
from mpd_df.models import make_psd_svm, make_random_forest
from mpd_df.splits import loso_splits, within_subject_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task", choices=("A", "B", "C"), required=True)
    parser.add_argument("--model", choices=("psd_svm", "random_forest"), required=True)
    parser.add_argument("--protocol", choices=("within_subject", "loso"), required=True)
    parser.add_argument("--subject-allowlist", type=Path)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args()


def load_features(
    root: Path,
    expected_task: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task") != expected_task:
        raise ValueError(
            f"Feature cache task {manifest.get('task')!r} does not match requested {expected_task!r}"
        )
    expected_subjects = set(manifest.get("subjects", []))
    expected_fingerprints = manifest.get("subject_fingerprints", {})
    arrays = []
    targets = []
    metadata = []
    observed_subjects = set()
    expected_feature_names = None
    for path in sorted(root.glob("subject_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            subject_id = path.stem.removeprefix("subject_")
            observed_subjects.add(subject_id)
            observed_fingerprint = str(data["cache_fingerprint"].item())
            if observed_fingerprint != expected_fingerprints.get(subject_id):
                raise ValueError(f"Fingerprint mismatch in {path.name}")
            if str(data["config_fingerprint"].item()) != manifest.get("config_fingerprint"):
                raise ValueError(f"Configuration fingerprint mismatch in {path.name}")
            feature_names = data["feature_names"].astype(str).tolist()
            if expected_feature_names is None:
                expected_feature_names = feature_names
            elif feature_names != expected_feature_names:
                raise ValueError(f"Feature schema mismatch in {path.name}")
            if not (
                len(data["X"])
                == len(data["y"])
                == len(data["subject"])
                == len(data["window_start_sec"])
                == len(data["block_id"])
                == len(data["group_id"])
            ):
                raise ValueError(f"Array length mismatch in {path.name}")
            arrays.append(data["X"])
            targets.append(data["y"])
            metadata.append(
                pd.DataFrame(
                    {
                        "subject": data["subject"].astype(str),
                        "window_start_sec": data["window_start_sec"],
                        "block_id": data["block_id"],
                        "group_id": data["group_id"].astype(str),
                    }
                )
            )
    if not arrays:
        raise FileNotFoundError(f"No subject_*.npz files found in {root}")
    if observed_subjects != expected_subjects:
        raise ValueError(
            f"Cache roster mismatch; missing={sorted(expected_subjects-observed_subjects)}, "
            f"extra={sorted(observed_subjects-expected_subjects)}"
        )
    return (
        np.concatenate(arrays),
        np.concatenate(targets),
        pd.concat(metadata, ignore_index=True),
        manifest,
    )


def build_split_plan(
    metadata: pd.DataFrame,
    y: np.ndarray,
    protocol: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[dict[str, str]]]:
    exclusions: list[dict[str, str]] = []
    if protocol == "loso":
        selected = np.arange(len(metadata))
        split_list = list(loso_splits(metadata))
    else:
        eligible_indices = []
        local_plans = []
        for subject in metadata["subject"].unique():
            subject_indices = np.flatnonzero(metadata["subject"].to_numpy() == subject)
            local = metadata.iloc[subject_indices].reset_index(drop=True)
            try:
                local_splits = list(within_subject_splits(local, y[subject_indices], seed=seed))
            except ValueError as exc:
                exclusions.append({"subject": str(subject), "reason": str(exc)})
                continue
            base = len(eligible_indices)
            eligible_indices.extend(subject_indices.tolist())
            local_plans.extend(
                (np.asarray(train) + base, np.asarray(test) + base)
                for train, test in local_splits
            )
        selected = np.asarray(eligible_indices, dtype=int)
        split_list = local_plans
        metadata = metadata.iloc[selected].reset_index(drop=True)
    return metadata, selected, split_list, exclusions


def main() -> None:
    args = parse_args()
    X, y, metadata, manifest = load_features(args.feature_dir, args.task)
    input_subjects = sorted(metadata["subject"].astype(str).str.zfill(2).unique())
    allowlist_excluded: list[str] = []
    if args.subject_allowlist is not None:
        allowed = {
            value.strip().zfill(2)
            for value in args.subject_allowlist.read_text().splitlines()
            if value.strip()
        }
        selected = metadata["subject"].astype(str).str.zfill(2).isin(allowed).to_numpy()
        X = X[selected]
        y = y[selected]
        metadata = metadata.loc[selected].reset_index(drop=True)
        allowlist_excluded = sorted(set(input_subjects) - allowed)
        if metadata.empty:
            raise ValueError("Subject allowlist removed every cached window")
    factory = (
        (lambda: make_psd_svm(seed=args.seed))
        if args.model == "psd_svm"
        else (lambda: make_random_forest(seed=args.seed))
    )
    metadata, selected, split_list, exclusions = build_split_plan(
        metadata,
        y,
        args.protocol,
        args.seed,
    )
    X = X[selected]
    y = y[selected]
    if not split_list:
        raise ValueError("No valid folds remain after eligibility checks")
    command = " ".join(shlex.quote(value) for value in sys.argv)
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    context = ExperimentContext(
        experiment_id=args.output_dir.name,
        task=args.task,
        model=args.model,
        protocol=args.protocol,
        seed=args.seed,
    )
    output = initialize_experiment(args.output_dir, context, config, command, args.project_root)
    (output / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pd.DataFrame(exclusions, columns=["subject", "reason"]).to_csv(
        output / "excluded_subjects.csv",
        index=False,
    )
    assignments = []
    for fold, (train, test) in enumerate(split_list):
        assignments.append(
            {
                "fold": fold,
                "train_subjects": ";".join(sorted(metadata.iloc[train]["subject"].unique())),
                "test_subjects": ";".join(sorted(metadata.iloc[test]["subject"].unique())),
                "train_groups": ";".join(sorted(metadata.iloc[train]["group_id"].unique())),
                "test_groups": ";".join(sorted(metadata.iloc[test]["group_id"].unique())),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    assignment_frame = pd.DataFrame(assignments)
    assignment_frame.to_csv(output / "split_assignments.csv", index=False)
    folds, predictions, pooled = evaluate_classical_splits(
        X,
        y,
        metadata,
        iter(split_list),
        factory,
    )
    metrics = binary_metrics(y, pooled)
    save_experiment_results(output, folds, predictions, metrics)
    metrics_path = output / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text())
    metrics_payload["eligibility"] = {
        "n_subjects_input": len(input_subjects),
        "n_subjects_considered": metadata["subject"].nunique(),
        "n_subjects_evaluated": predictions["subject"].nunique(),
        "n_subjects_allowlist_excluded": len(allowlist_excluded),
        "subjects_allowlist_excluded": allowlist_excluded,
        "subjects_protocol_excluded": sorted(
            {str(row["subject"]) for row in exclusions}
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False) + "\n")
    (output / "subjects_train.txt").write_text(
        "\n".join(
            f"fold={row.fold}: {row.train_subjects}"
            for row in assignment_frame.itertuples()
        )
        + "\n"
    )
    (output / "subjects_validation.txt").write_text("Not used by this fixed baseline.\n")
    (output / "subjects_test.txt").write_text(
        "\n".join(
            f"fold={row.fold}: {row.test_subjects}"
            for row in assignment_frame.itertuples()
        )
        + "\n"
    )
    (output / "README.md").write_text(
        f"# {context.experiment_id}\n\n"
        f"Task `{args.task}`, model `{args.model}`, protocol `{args.protocol}`.\n"
    )
    final = finalize_experiment(output, args.output_dir)
    print(json.dumps({"output": str(final), "micro_window_pooled": metrics}, indent=2))


if __name__ == "__main__":
    main()
