"""Memory-mapped EEG caches and deterministic PyTorch training helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .splits import loso_splits, within_subject_splits


@dataclass
class WindowCache:
    root: Path
    manifest: dict[str, object]
    metadata: pd.DataFrame
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, root: str | Path) -> "WindowCache":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        frames = []
        arrays = {}
        for record in manifest["subjects"]:
            subject = str(record["subject"]).zfill(2)
            array = np.load(root / record["data"], mmap_mode="r")
            frame = pd.read_csv(
                root / record["metadata"],
                dtype={"subject": str, "group_id": str},
            )
            if len(array) != len(frame):
                raise ValueError(f"Cache length mismatch for subject {subject}")
            frame["subject"] = frame["subject"].str.zfill(2)
            frame["_cache_subject"] = subject
            frame["_cache_row"] = np.arange(len(frame))
            arrays[subject] = array
            frames.append(frame)
        if not frames:
            raise ValueError("Deep cache contains no subjects")
        return cls(root, manifest, pd.concat(frames, ignore_index=True), arrays)


class CachedWindowDataset(Dataset):
    def __init__(self, cache: WindowCache) -> None:
        self.cache = cache

    def __len__(self) -> int:
        return len(self.cache.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.cache.metadata.iloc[index]
        values = np.asarray(
            self.cache.arrays[row["_cache_subject"]][int(row["_cache_row"])],
            dtype=np.float32,
        ).copy()
        return torch.from_numpy(values), torch.tensor(int(row["target"]), dtype=torch.long)


def validation_split(
    train_indices: np.ndarray,
    metadata: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    local = metadata.iloc[train_indices]
    groups = (
        local["subject"].to_numpy()
        if local["subject"].nunique() > 1
        else local["group_id"].to_numpy()
    )
    class_group_counts = [
        np.unique(groups[y[train_indices] == class_id]).size for class_id in (0, 1)
    ]
    n_splits = min(5, *class_group_counts)
    if n_splits < 2:
        raise ValueError("Training fold lacks two independent groups per class")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_train, validation = next(
        splitter.split(np.zeros(len(train_indices)), y[train_indices], groups)
    )
    return train_indices[inner_train], train_indices[validation]


def build_deep_split_plan(
    metadata: pd.DataFrame,
    y: np.ndarray,
    protocol: str,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[dict[str, object]]]:
    """Preflight every outer and inner split before any model is trained."""

    plans: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    exclusions: list[dict[str, object]] = []
    if protocol == "loso":
        for fold, (outer_train, test) in enumerate(loso_splits(metadata)):
            try:
                train, validation = validation_split(
                    outer_train,
                    metadata,
                    y,
                    seed + fold,
                )
            except ValueError as exc:
                exclusions.append(
                    {
                        "scope": "fold",
                        "subject": ";".join(sorted(metadata.iloc[test]["subject"].unique())),
                        "fold": fold,
                        "reason": str(exc),
                    }
                )
                continue
            plans.append((train, validation, test))
        return plans, exclusions

    if protocol != "within_subject":
        raise ValueError(f"Unsupported protocol: {protocol}")
    for subject in metadata["subject"].unique():
        indices = np.flatnonzero(metadata["subject"].to_numpy() == subject)
        local = metadata.iloc[indices].reset_index(drop=True)
        try:
            outer = list(within_subject_splits(local, y[indices], seed=seed))
            subject_plans = []
            for fold, (local_train, local_test) in enumerate(outer):
                outer_train = indices[local_train]
                test = indices[local_test]
                train, validation = validation_split(
                    outer_train,
                    metadata,
                    y,
                    seed + fold,
                )
                subject_plans.append((train, validation, test))
        except ValueError as exc:
            exclusions.append(
                {
                    "scope": "subject",
                    "subject": str(subject),
                    "fold": None,
                    "reason": str(exc),
                }
            )
            continue
        plans.extend(subject_plans)
    return plans, exclusions


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(y: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y[indices], minlength=2).astype(float)
    if np.any(counts == 0):
        raise ValueError("Both classes are required in every training fold")
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets = []
    predictions = []
    scores = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for values, target in loader:
            values = values.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            probability = torch.softmax(logits.detach(), dim=1)[:, 1]
            targets.append(target.detach().cpu().numpy())
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            scores.append(probability.cpu().numpy())
    size = max(1, sum(len(values) for values in targets))
    return (
        total_loss / size,
        np.concatenate(targets),
        np.concatenate(predictions),
        np.concatenate(scores),
    )
