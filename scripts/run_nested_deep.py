#!/usr/bin/env python3
"""Nested EEGNet evaluation for final binary EEG tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from mpd_df.deep_training import (
    CachedWindowDataset,
    WindowCache,
    class_weights,
    choose_device,
    make_loader,
    run_epoch,
    validation_split,
)
from mpd_df.experiment import (
    ExperimentContext,
    finalize_experiment,
    initialize_experiment,
    save_experiment_results,
)
from mpd_df.metrics import binary_metrics
from mpd_df.models import EEGNet
from mpd_df.nested import best_epoch_count, select_threshold
from mpd_df.reproducibility import set_global_seed
from mpd_df.splits import loso_splits, within_subject_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("within_subject", "loso"), required=True)
    parser.add_argument("--task", choices=("A", "B"), required=True)
    parser.add_argument("--subject-allowlist", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--selection-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def candidates() -> list[dict[str, float | int]]:
    return [
        {"eegnet_f1": 8, "dropout": 0.25, "learning_rate": 1e-3},
        {"eegnet_f1": 8, "dropout": 0.50, "learning_rate": 3e-4},
        {"eegnet_f1": 16, "dropout": 0.25, "learning_rate": 3e-4},
        {"eegnet_f1": 16, "dropout": 0.50, "learning_rate": 1e-3},
    ]


def model_for(params: dict[str, float | int], channels: int, times: int) -> EEGNet:
    f1 = int(params["eegnet_f1"])
    return EEGNet(
        n_channels=channels,
        n_times=times,
        f1=f1,
        depth_multiplier=2,
        f2=2 * f1,
        temporal_kernel=64,
        dropout=float(params["dropout"]),
    )


def fit_fixed(
    dataset: CachedWindowDataset,
    metadata: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    params: dict[str, float | int],
    device: torch.device,
    channels: int,
    times: int,
    args: argparse.Namespace,
    epochs: int,
    early_stop: bool = True,
    evaluate_each_epoch: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], dict[str, torch.Tensor]]:
    set_global_seed(args.seed)
    model = model_for(params, channels, times).to(device)
    loss = torch.nn.CrossEntropyLoss(class_weights(y, train, device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["learning_rate"]), weight_decay=1e-4
    )
    train_loader = make_loader(
        dataset, metadata.iloc[train]._dataset_index, args.batch_size, True,
        args.seed, args.num_workers,
    )
    valid_loader = make_loader(
        dataset, metadata.iloc[valid]._dataset_index, args.batch_size, False,
        args.seed, args.num_workers,
    )
    history = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(epochs):
        train_loss, _, _, _ = run_epoch(model, train_loader, loss, device, optimizer)
        valid_loss = np.nan
        if evaluate_each_epoch:
            valid_loss, truth, _, scores = run_epoch(
                model, valid_loader, loss, device, None
            )
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if early_stop and stale >= args.patience:
                    history.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": valid_loss,
                    })
                    break
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
        })
    if best_state is not None:
        model.load_state_dict(best_state)
    _, truth, _, scores = run_epoch(model, valid_loader, loss, device, None)
    final_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return truth, scores, history, final_state


def outer_splits(metadata: pd.DataFrame, y: np.ndarray, protocol: str):
    if protocol == "loso":
        yield from loso_splits(metadata)
        return
    subjects = metadata.subject.to_numpy()
    for subject in metadata.subject.unique():
        indices = np.flatnonzero(subjects == subject)
        local = metadata.iloc[indices].reset_index(drop=True)
        for train, test in within_subject_splits(local, y[indices], seed=42):
            yield indices[train], indices[test]


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    cache = WindowCache.load(args.cache_dir)
    if cache.manifest["task"] != args.task:
        raise ValueError("Deep cache task does not match --task")
    dataset = CachedWindowDataset(cache)
    metadata = cache.metadata.copy()
    metadata["_dataset_index"] = np.arange(len(metadata))
    if args.subject_allowlist:
        allowed = {v.strip().zfill(2) for v in args.subject_allowlist.read_text().splitlines() if v.strip()}
        metadata = metadata[metadata.subject.astype(str).str.zfill(2).isin(allowed)].reset_index(drop=True)
    y = metadata.target.to_numpy(np.int8)
    records = cache.manifest["subjects"]
    channels, times = int(records[0]["n_channels"]), int(records[0]["n_times"])
    device = choose_device(args.device)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    partial = initialize_experiment(
        args.output_dir,
        ExperimentContext(
            args.output_dir.name,
            args.task,
            "eegnet",
            args.protocol,
            args.seed,
        ),
        resolved,
        " ".join(shlex.quote(v) for v in sys.argv),
        Path("."),
    )
    predictions, fold_rows, selection_rows, histories, assignments = [], [], [], [], []
    (partial / "checkpoints").mkdir()
    all_outer_splits = list(outer_splits(metadata, y, args.protocol))
    total_folds = min(len(all_outer_splits), args.max_folds or len(all_outer_splits))
    for fold, (outer_train, test) in enumerate(all_outer_splits):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        print(
            f"[{args.protocol}/eegnet] outer fold {fold + 1}/{total_folds}",
            flush=True,
        )
        groups = (
            metadata.iloc[outer_train].subject.to_numpy()
            if args.protocol == "loso"
            else metadata.iloc[outer_train].group_id.to_numpy()
        )
        local_y = y[outer_train]
        n_splits = min(5, *(np.unique(groups[local_y == cls]).size for cls in (0, 1)))
        if n_splits < 2:
            raise ValueError(
                f"Fold {fold} has fewer than two independent groups per class"
            )
        inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + fold)
        best = None
        for candidate_id, candidate in enumerate(candidates()):
            print(
                f"[{args.protocol}/eegnet] fold {fold + 1}: "
                f"candidate {candidate_id + 1}/{len(candidates())}",
                flush=True,
            )
            pooled_truth, pooled_scores = [], []
            for inner_train, inner_valid in inner.split(np.zeros(len(outer_train)), local_y, groups):
                truth, scores, _, _ = fit_fixed(
                    dataset, metadata, y, outer_train[inner_train], outer_train[inner_valid],
                    candidate, device, channels, times, args, args.selection_epochs,
                )
                pooled_truth.append(truth); pooled_scores.append(scores)
            truth = np.concatenate(pooled_truth); scores = np.concatenate(pooled_scores)
            threshold, metrics = select_threshold(truth, scores)
            rank = tuple(float(metrics.get(k) or -1) for k in ("f1", "recall", "kappa", "balanced_accuracy"))
            selection_rows.append({"fold": fold, "candidate_id": candidate_id, **candidate, "threshold": threshold, **metrics})
            if best is None or rank > best[0]:
                best = (rank, candidate, threshold)
        assert best is not None
        selected, threshold = best[1], best[2]
        train, validation = validation_split(outer_train, metadata, y, args.seed + fold)
        _, _, history, _ = fit_fixed(
            dataset, metadata, y, train, validation, selected, device,
            channels, times, args, args.epochs,
        )
        selected_epochs = best_epoch_count(history)
        # Refit once on outer training for the selected epoch budget.
        test_truth, test_scores, _, final_state = fit_fixed(
            dataset, metadata, y, outer_train, test, selected, device,
            channels, times, args, selected_epochs, early_stop=False,
            evaluate_each_epoch=False,
        )
        torch.save(final_state, partial / "checkpoints" / f"fold_{fold:03d}.pt")
        predicted = (test_scores >= threshold).astype(np.int8)
        metrics = binary_metrics(test_truth, predicted); matrix = metrics.pop("confusion_matrix")
        fold_rows.append({
            "fold": fold,
            **selected,
            "selected_epochs": selected_epochs,
            "threshold": threshold,
            **metrics,
            "confusion_matrix": json.dumps(matrix),
        })
        rows = metadata.iloc[test].drop(columns="_dataset_index").copy()
        rows["fold"] = fold; rows["y_true"] = test_truth; rows["y_pred"] = predicted; rows["score"] = test_scores
        predictions.append(rows)
        histories.extend({"fold": fold, **row} for row in history)
        assignments.append({
            "fold": fold,
            "train_subjects": ";".join(sorted(metadata.iloc[outer_train].subject.unique())),
            "test_subjects": ";".join(sorted(metadata.iloc[test].subject.unique())),
            "train_groups": ";".join(sorted(metadata.iloc[outer_train].group_id.unique())),
            "test_groups": ";".join(sorted(metadata.iloc[test].group_id.unique())),
        })
    prediction_frame = pd.concat(predictions, ignore_index=True)
    save_experiment_results(
        partial, pd.DataFrame(fold_rows), prediction_frame,
        binary_metrics(prediction_frame.y_true.to_numpy(), prediction_frame.y_pred.to_numpy()),
    )
    pd.DataFrame(selection_rows).to_csv(partial / "inner_selection.csv", index=False)
    pd.DataFrame(histories).to_csv(partial / "learning_curves.csv", index=False)
    pd.DataFrame(assignments).to_csv(partial / "split_assignments.csv", index=False)
    (partial / "device.json").write_text(json.dumps({
        "device": str(device), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2) + "\n")
    for filename in ("subjects_train.txt", "subjects_validation.txt", "subjects_test.txt"):
        (partial / filename).write_text("See split_assignments.csv\n")
    (partial / "README.md").write_text(
        f"# Final nested EEGNet Task {args.task}\n"
    )
    finalize_experiment(partial, args.output_dir)
    print(f"[{args.protocol}/eegnet] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
