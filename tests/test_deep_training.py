import json

import numpy as np
import pandas as pd
import torch

from mpd_df.deep_training import (
    CachedWindowDataset,
    WindowCache,
    build_deep_split_plan,
    choose_device,
    validation_split,
)


def test_memory_mapped_window_cache(tmp_path) -> None:
    values = np.arange(8 * 2 * 10, dtype=np.float32).reshape(8, 2, 10)
    np.save(tmp_path / "subject_01_X.npy", values)
    metadata = pd.DataFrame(
        {
            "subject": ["01"] * 8,
            "group_id": [f"01:{index // 2}" for index in range(8)],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    metadata.to_csv(tmp_path / "subject_01_metadata.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "task": "C",
                "subjects": [
                    {
                        "subject": "01",
                        "data": "subject_01_X.npy",
                        "metadata": "subject_01_metadata.csv",
                    }
                ],
            }
        )
    )

    cache = WindowCache.load(tmp_path)
    dataset = CachedWindowDataset(cache)
    signal, target = dataset[6]

    assert signal.shape == (2, 10)
    assert target.item() == 1
    assert torch.isfinite(signal).all()


def test_validation_split_keeps_groups_disjoint() -> None:
    metadata = pd.DataFrame(
        {
            "subject": ["01"] * 16,
            "group_id": [f"01:{index // 2}" for index in range(16)],
        }
    )
    y = np.array([0] * 8 + [1] * 8)
    train, validation = validation_split(np.arange(16), metadata, y, seed=42)

    assert set(metadata.iloc[train]["group_id"]).isdisjoint(
        set(metadata.iloc[validation]["group_id"])
    )


def test_choose_device_cpu() -> None:
    assert choose_device("cpu").type == "cpu"


def test_split_preflight_records_ineligible_subject() -> None:
    rows = []
    targets = []
    for subject, groups_per_class in (("01", 4), ("02", 2)):
        for target in (0, 1):
            for group in range(groups_per_class):
                for _ in range(2):
                    rows.append(
                        {
                            "subject": subject,
                            "group_id": f"{subject}:{target}:{group}",
                        }
                    )
                    targets.append(target)
    metadata = pd.DataFrame(rows)

    plans, exclusions = build_deep_split_plan(
        metadata,
        np.asarray(targets),
        protocol="within_subject",
        seed=42,
    )

    assert len(plans) == 4
    assert exclusions == [
        {
            "scope": "subject",
            "subject": "02",
            "fold": None,
            "reason": "Training fold lacks two independent groups per class",
        }
    ]
