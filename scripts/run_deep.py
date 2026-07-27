#!/usr/bin/env python3
"""Train EEGNet or the explicitly inferred local MSCNN-CAM implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd
import torch

from mpd_df.constants import GLOBAL_SEED
from mpd_df.deep_training import (
    CachedWindowDataset,
    WindowCache,
    build_deep_split_plan,
    choose_device,
    class_weights,
    make_loader,
    run_epoch,
)
from mpd_df.experiment import (
    ExperimentContext,
    finalize_experiment,
    initialize_experiment,
    save_experiment_results,
)
from mpd_df.metrics import binary_metrics
from mpd_df.models import EEGNet, MSCNNCAM
from mpd_df.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task", choices=("A", "B", "C"), required=True)
    parser.add_argument("--model", choices=("eegnet", "mscnn_cam"), required=True)
    parser.add_argument("--protocol", choices=("within_subject", "loso"), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eegnet-f1", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel", type=int, default=64)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--subject-allowlist", type=Path)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args()


def make_model(args: argparse.Namespace, n_channels: int, n_times: int):
    name = args.model
    if name == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            f1=args.eegnet_f1,
            depth_multiplier=args.eegnet_depth_multiplier,
            f2=args.eegnet_f2,
            temporal_kernel=args.eegnet_temporal_kernel,
            dropout=args.eegnet_dropout,
        )
    return MSCNNCAM(n_channels=n_channels)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    set_global_seed(args.seed)
    cache = WindowCache.load(args.cache_dir)
    if cache.manifest["task"] != args.task:
        raise ValueError("Cache task does not match requested task")
    dataset = CachedWindowDataset(cache)
    metadata = cache.metadata
    input_subjects = sorted(metadata["subject"].astype(str).str.zfill(2).unique())
    allowlist_excluded: list[str] = []
    if args.subject_allowlist is not None:
        allowed = {
            value.strip().zfill(2)
            for value in args.subject_allowlist.read_text().splitlines()
            if value.strip()
        }
        metadata = metadata.loc[
            metadata["subject"].astype(str).str.zfill(2).isin(allowed)
        ].copy()
        allowlist_excluded = sorted(set(input_subjects) - allowed)
        if metadata.empty:
            raise ValueError("Subject allowlist removed every cached window")
        metadata["_dataset_index"] = metadata.index
        metadata = metadata.reset_index(drop=True)
    else:
        metadata = metadata.copy()
        metadata["_dataset_index"] = metadata.index
    y = metadata["target"].to_numpy(dtype=np.int8)
    device = choose_device(args.device)
    records = cache.manifest["subjects"]
    n_channels = int(records[0]["n_channels"])
    n_times = int(records[0]["n_times"])
    command = " ".join(shlex.quote(value) for value in sys.argv)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["resolved_device"] = str(device)
    config["implementation_status"] = (
        "standard_eegnet_local"
        if args.model == "eegnet"
        else "inferred_local_mscnn_cam_not_official_table9_code"
    )
    context = ExperimentContext(
        experiment_id=args.output_dir.name,
        task=args.task,
        model=args.model,
        protocol=args.protocol,
        seed=args.seed,
    )
    split_plan, exclusions = build_deep_split_plan(
        metadata,
        y,
        args.protocol,
        args.seed,
    )
    if not split_plan:
        raise ValueError("No valid deep-learning folds remain after split preflight")
    partial = initialize_experiment(
        args.output_dir,
        context,
        config,
        command,
        args.project_root,
    )
    (partial / "checkpoints").mkdir()
    fold_rows = []
    prediction_frames = []
    learning_rows = []
    assignment_rows = []
    pd.DataFrame(exclusions, columns=["scope", "subject", "fold", "reason"]).to_csv(
        partial / "excluded_subjects.csv",
        index=False,
    )

    for fold, (train, validation, test) in enumerate(split_plan):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        assignment_rows.append(
            {
                "fold": fold,
                "train_subjects": ";".join(sorted(metadata.iloc[train]["subject"].unique())),
                "validation_subjects": ";".join(
                    sorted(metadata.iloc[validation]["subject"].unique())
                ),
                "test_subjects": ";".join(sorted(metadata.iloc[test]["subject"].unique())),
                "train_groups": ";".join(sorted(metadata.iloc[train]["group_id"].unique())),
                "validation_groups": ";".join(
                    sorted(metadata.iloc[validation]["group_id"].unique())
                ),
                "test_groups": ";".join(sorted(metadata.iloc[test]["group_id"].unique())),
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
            }
        )
        model = make_model(args, n_channels, n_times).to(device)
        criterion = torch.nn.CrossEntropyLoss(class_weights(y, train, device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        dataset_train = metadata.iloc[train]["_dataset_index"].to_numpy()
        dataset_validation = metadata.iloc[validation]["_dataset_index"].to_numpy()
        dataset_test = metadata.iloc[test]["_dataset_index"].to_numpy()
        train_loader = make_loader(
            dataset,
            dataset_train,
            args.batch_size,
            True,
            args.seed + fold,
            args.num_workers,
        )
        validation_loader = make_loader(
            dataset,
            dataset_validation,
            args.batch_size,
            False,
            args.seed,
            args.num_workers,
        )
        checkpoint = partial / "checkpoints" / f"fold_{fold:02d}.pt"
        best_loss = float("inf")
        stale = 0
        for epoch in range(args.epochs):
            train_loss, _, _, _ = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            validation_loss, _, _, _ = run_epoch(
                model, validation_loader, criterion, device, None
            )
            learning_rows.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                stale = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test_loader = make_loader(
            dataset,
            dataset_test,
            args.batch_size,
            False,
            args.seed,
            args.num_workers,
        )
        test_loss, y_true, y_pred, score = run_epoch(
            model, test_loader, criterion, device, None
        )
        metrics = binary_metrics(y_true, y_pred)
        matrix = metrics.pop("confusion_matrix")
        fold_rows.append(
            {
                "fold": fold,
                "test_loss": test_loss,
                **metrics,
                "confusion_matrix": json.dumps(matrix),
            }
        )
        rows = metadata.iloc[test].drop(columns="_dataset_index").copy()
        rows["fold"] = fold
        rows["y_true"] = y_true
        rows["y_pred"] = y_pred
        rows["score"] = score
        prediction_frames.append(rows)

    if not prediction_frames:
        raise ValueError("No valid deep-learning folds were produced")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = binary_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
    )
    save_experiment_results(partial, pd.DataFrame(fold_rows), predictions, pooled)
    metrics_path = partial / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text())
    evaluated_subjects = sorted(predictions["subject"].astype(str).unique())
    considered_subjects = sorted(metadata["subject"].astype(str).unique())
    metrics_payload["eligibility"] = {
        "n_subjects_input": len(input_subjects),
        "n_subjects_considered": len(considered_subjects),
        "n_subjects_evaluated": len(evaluated_subjects),
        "n_subjects_nested_split_excluded": len(
            {str(row["subject"]) for row in exclusions if row["scope"] == "subject"}
        ),
        "n_subjects_allowlist_excluded": len(allowlist_excluded),
        "subjects_evaluated": evaluated_subjects,
        "subjects_nested_split_excluded": sorted(
            {str(row["subject"]) for row in exclusions if row["scope"] == "subject"}
        ),
        "subjects_allowlist_excluded": allowlist_excluded,
        "excluded_fold_count": sum(row["scope"] == "fold" for row in exclusions),
        "max_folds_smoke_limit": args.max_folds,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(learning_rows).to_csv(partial / "learning_curves.csv", index=False)
    assignments = pd.DataFrame(assignment_rows)
    assignments.to_csv(partial / "split_assignments.csv", index=False)
    for column, filename in (
        ("train_subjects", "subjects_train.txt"),
        ("validation_subjects", "subjects_validation.txt"),
        ("test_subjects", "subjects_test.txt"),
    ):
        (partial / filename).write_text(
            "\n".join(
                f"fold={row.fold}: {getattr(row, column)}"
                for row in assignments.itertuples()
            )
            + "\n"
        )
    (partial / "README.md").write_text(
        f"# {context.experiment_id}\n\n"
        f"Task `{args.task}`, model `{args.model}`, protocol `{args.protocol}`.\n\n"
        + (
            "This MSCNN-CAM is an inferred local reimplementation and must not be "
            "reported as the official Table 9 implementation.\n"
            if args.model == "mscnn_cam"
            else "EEGNet standardized local benchmark.\n"
        )
    )
    (partial / "device.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cudnn": torch.backends.cudnn.version(),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
            indent=2,
        )
        + "\n"
    )
    final = finalize_experiment(partial, args.output_dir)
    print(json.dumps({"output": str(final), "metrics": pooled}, indent=2))


if __name__ == "__main__":
    main()
