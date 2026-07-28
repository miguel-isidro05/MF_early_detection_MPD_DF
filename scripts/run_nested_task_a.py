#!/usr/bin/env python3
"""Nested binary-task executor for classical EEG baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import yaml

from mpd_df.experiment import ExperimentContext, finalize_experiment, initialize_experiment, save_experiment_results
from mpd_df.metrics import binary_metrics
from mpd_df.models import make_psd_svm, make_random_forest
from mpd_df.nested import nested_classical_selection
from mpd_df.splits import loso_splits, within_subject_splits


def candidate_grid(model: str) -> list[dict[str, object]]:
    if model == "psd_svm":
        return [{"c": value} for value in (0.1, 1.0, 10.0)]
    return [{"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (12, None) for leaf in (1, 5)]


def estimator(
    model: str,
    params: dict[str, object],
    seed: int,
    svm_solver: dict[str, object],
):
    if model == "psd_svm":
        return make_psd_svm(
            seed=seed,
            c=float(params["c"]),
            max_iter=int(svm_solver["max_iter"]),
            tol=float(svm_solver["tolerance"]),
        )
    return make_random_forest(seed=seed, max_depth=params["max_depth"], min_samples_leaf=int(params["min_samples_leaf"]))


def main() -> None:
    warnings.filterwarnings("error", category=ConvergenceWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("psd_svm", "random_forest"), required=True)
    parser.add_argument("--protocol", choices=("within_subject", "loso"), required=True)
    parser.add_argument("--task", choices=("A", "B"), required=True)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--subject-allowlist", type=Path)
    args = parser.parse_args()
    if not args.raw_root.is_dir():
        raise FileNotFoundError(args.raw_root)
    config = yaml.safe_load(args.config.read_text())
    if config.get("task") != args.task or config.get("models") != ["psd_svm", "random_forest", "eegnet"]:
        raise ValueError("Executor task/models do not match the declared protocol")
    svm_solver = config.get("svm_solver", {})
    required_solver_keys = {"max_iter", "tolerance", "convergence_warning"}
    if set(svm_solver) != required_solver_keys:
        raise ValueError("svm_solver config must define the complete solver policy")
    if svm_solver["convergence_warning"] != "fail_run":
        raise ValueError("Final protocol requires convergence_warning=fail_run")
    from run_classical import load_features
    X, y, metadata, _ = load_features(args.feature_dir, args.task)
    if args.subject_allowlist:
        allowed = {v.strip().zfill(2) for v in args.subject_allowlist.read_text().splitlines() if v.strip()}
        keep = metadata.subject.astype(str).str.zfill(2).isin(allowed).to_numpy()
        X, y, metadata = X[keep], y[keep], metadata.loc[keep].reset_index(drop=True)
    if args.protocol == "loso":
        outer = list(loso_splits(metadata))
    else:
        outer = []
        subject_values = metadata.subject.to_numpy()
        for subject in metadata.subject.unique():
            global_indices = np.flatnonzero(subject_values == subject)
            local = metadata.iloc[global_indices].reset_index(drop=True)
            for train, test in within_subject_splits(local, y[global_indices], seed=42):
                outer.append((global_indices[train], global_indices[test]))
    output_dir = args.output_root / args.protocol / args.model
    context = ExperimentContext(output_dir.name, args.task, args.model, args.protocol, 42)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    partial = initialize_experiment(output_dir, context, resolved, " ".join(__import__('sys').argv), Path("."))
    predictions, folds, selections, assignments = [], [], [], []
    for fold, (train, test) in enumerate(outer):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        print(
            f"[{args.protocol}/{args.model}] outer fold {fold + 1}/"
            f"{min(len(outer), args.max_folds or len(outer))}",
            flush=True,
        )
        inner_groups = (metadata.iloc[train].subject.to_numpy() if args.protocol == "loso"
                        else metadata.iloc[train].group_id.to_numpy())
        selected, threshold, inner = nested_classical_selection(
            X[train], y[train], inner_groups, candidate_grid(args.model),
            lambda params: estimator(
                args.model,
                params,
                42 + fold,
                svm_solver,
            ),
            42 + fold,
        )
        model = estimator(args.model, selected, 42 + fold, svm_solver)
        model.fit(X[train], y[train])
        score = model.decision_function(X[test]) if hasattr(model, "decision_function") else model.predict_proba(X[test])[:, 1]
        pred = (score >= threshold).astype(np.int8); metric = binary_metrics(y[test], pred); matrix = metric.pop("confusion_matrix")
        folds.append({
            "fold": fold,
            "selected_params": json.dumps(selected, sort_keys=True),
            "threshold": threshold,
            **metric,
            "confusion_matrix": json.dumps(matrix),
        })
        frame = metadata.iloc[test].copy(); frame["fold"] = fold; frame["y_true"] = y[test]; frame["y_pred"] = pred; frame["score"] = score; predictions.append(frame)
        selections.extend({"fold": fold, **row} for row in inner)
        assignments.append({
            "fold": fold,
            "train_subjects": ";".join(sorted(metadata.iloc[train].subject.unique())),
            "test_subjects": ";".join(sorted(metadata.iloc[test].subject.unique())),
            "train_groups": ";".join(sorted(metadata.iloc[train].group_id.unique())),
            "test_groups": ";".join(sorted(metadata.iloc[test].group_id.unique())),
        })
    prediction_frame = pd.concat(predictions, ignore_index=True)
    save_experiment_results(partial, pd.DataFrame(folds), prediction_frame, binary_metrics(prediction_frame.y_true.to_numpy(), prediction_frame.y_pred.to_numpy()))
    pd.DataFrame(selections).to_json(partial / "inner_selection.json", orient="records", indent=2)
    pd.DataFrame(assignments).to_csv(partial / "split_assignments.csv", index=False)
    (partial / "subjects_train.txt").write_text("See split_assignments.csv\n")
    (partial / "subjects_validation.txt").write_text("Inner grouped validation; see inner_selection.json\n")
    (partial / "subjects_test.txt").write_text("See split_assignments.csv\n")
    (partial / "README.md").write_text(
        f"# Nested Task {args.task} {args.model} {args.protocol}\n"
    )
    finalize_experiment(partial, output_dir)
    print(f"[{args.protocol}/{args.model}] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
