"""Project-wide constants with scientific meaning."""

from __future__ import annotations

GLOBAL_SEED = 42

LABEL_NAMES = {
    0: "Wakefulness",
    1: "Fatigue1",
    2: "Fatigue2",
    3: "Fatigue3",
    4: "Fatigue4",
    8: "Signal Abnormality",
    9: "Severe Artifacts",
}

ARTIFACT_LABELS = frozenset({8, 9})

TASKS = {
    "A": {"negative": frozenset({0}), "positive": frozenset({1})},
    "B": {"negative": frozenset({0}), "positive": frozenset({1, 2})},
    "C": {"negative": frozenset({0}), "positive": frozenset({1, 2, 3, 4})},
}

# The EDF files contain 32 channels. The descriptor's analysis list omits the
# four midline channels below; both montages remain explicit until reproduction
# determines which one was used for Table 9.
EDF_CHANNELS = (
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "Cz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
)

PAPER_ANALYTICAL_CHANNELS = tuple(
    channel for channel in EDF_CHANNELS if channel not in {"Fz", "Pz", "Oz", "Cz"}
)

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

