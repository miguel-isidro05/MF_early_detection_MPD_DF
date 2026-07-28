#!/usr/bin/env python3
"""Generate subject-level physiological figures for Task B."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mpd_df.constants import BANDS
from mpd_df.statistics import holm_adjust
from run_classical import load_features


REGIONS = {
    "frontal": {"Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"},
    "frontocentral": {"FC1", "FC2", "FC5", "FC6", "FT9", "FT10"},
    "central": {"C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6"},
    "temporal": {"T7", "T8", "TP9", "TP10"},
    "parietal": {"P3", "P4", "P7", "P8", "Pz"},
    "occipital": {"O1", "O2", "Oz"},
}
LABEL_NAMES = {0: "Wakefulness", 1: "Fatigue1", 2: "Fatigue2"}
LABEL_COLORS = {0: "#277da1", 1: "#f8961e", 2: "#d62828"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


def parse_feature(name: str) -> tuple[str, str, str] | None:
    for kind in ("abs_db", "rel"):
        marker = f"_{kind}_"
        if marker in name:
            channel, band = name.split(marker, maxsplit=1)
            if band in BANDS:
                return channel, kind, band
    return None


def region_for(channel: str) -> str:
    matches = [region for region, channels in REGIONS.items() if channel in channels]
    if len(matches) != 1:
        raise ValueError(f"Channel {channel!r} has no unique region")
    return matches[0]


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return np.nan, np.nan
    samples = rng.choice(clean, size=(repetitions, len(clean)), replace=True)
    return tuple(np.quantile(samples.mean(axis=1), [0.025, 0.975]))


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=300)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def load_subject_summaries(
    feature_dir: Path,
) -> tuple[pd.DataFrame, list[str], list[tuple[str, str, str] | None]]:
    summaries: list[pd.DataFrame] = []
    X, _, metadata, _ = load_features(feature_dir, "B")
    if "label" not in metadata:
        raise ValueError(
            "Feature cache predates original-label export; rebuild the cache"
        )
    first_path = next(iter(sorted(feature_dir.glob("subject_*.npz"))), None)
    if first_path is None:
        raise FileNotFoundError(f"No subject_*.npz under {feature_dir}")
    with np.load(first_path, allow_pickle=False) as data:
        expected_names = data["feature_names"].astype(str).tolist()
    parsed_features = [parse_feature(name) for name in expected_names]
    for subject, subject_rows in metadata.groupby("subject", sort=True):
        subject_indices = subject_rows.index.to_numpy()
        labels = subject_rows["label"].to_numpy(dtype=np.int8)
        for label in (0, 1, 2):
            selected = labels == label
            if not selected.any():
                continue
            summaries.append(
                pd.DataFrame(
                    {
                        "subject": str(subject).zfill(2),
                        "label": label,
                        "feature_index": np.arange(X.shape[1]),
                        "value": X[subject_indices[selected]].mean(axis=0),
                        "n_windows": int(selected.sum()),
                    }
                )
            )
    summary = pd.concat(summaries, ignore_index=True)
    return summary, expected_names, parsed_features


def paired_effects(
    summary: pd.DataFrame,
    names: list[str],
    parsed: list[tuple[str, str, str] | None],
    seed: int,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    differences = []
    rng = np.random.default_rng(seed)
    for feature_index, feature in enumerate(parsed):
        if feature is None:
            continue
        channel, kind, band = feature
        local = summary[summary.feature_index == feature_index]
        wake = (
            local[local.label == 0]
            .set_index("subject")["value"]
            .rename("wake")
        )
        fatigue_rows = local[local.label.isin([1, 2])].copy()
        fatigue_rows["weighted_value"] = (
            fatigue_rows["value"] * fatigue_rows["n_windows"]
        )
        fatigue = (
            fatigue_rows.groupby("subject")
            .agg(
                weighted_sum=("weighted_value", "sum"),
                total_windows=("n_windows", "sum"),
            )
            .eval("weighted_sum / total_windows")
            .rename("fatigue")
        )
        paired = pd.concat([wake, fatigue], axis=1).dropna()
        delta = (paired["fatigue"] - paired["wake"]).to_numpy()
        if len(delta) < 2:
            continue
        std = float(np.std(delta, ddof=1))
        dz = float(np.mean(delta) / std) if std > 0 else np.nan
        low, high = bootstrap_ci(delta, rng, repetitions)
        if np.allclose(delta, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(delta)
        rows.append(
            {
                "feature": names[feature_index],
                "channel": channel,
                "region": region_for(channel),
                "kind": kind,
                "band": band,
                "n_subjects": len(delta),
                "wake_mean": float(paired["wake"].mean()),
                "fatigue12_mean": float(paired["fatigue"].mean()),
                "paired_difference_mean": float(np.mean(delta)),
                "paired_difference_ci95_low": low,
                "paired_difference_ci95_high": high,
                "cohens_dz": dz,
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
        differences.extend(
            {
                "subject": subject,
                "feature": names[feature_index],
                "channel": channel,
                "region": region_for(channel),
                "kind": kind,
                "band": band,
                "fatigue12_minus_wake": value,
            }
            for subject, value in zip(paired.index, delta)
        )
    effects = pd.DataFrame(rows)
    effects["p_holm"] = np.nan
    for kind, indices in effects.groupby("kind").groups.items():
        effects.loc[indices, "p_holm"] = holm_adjust(
            effects.loc[indices, "p_value"].to_numpy()
        )
    return effects, pd.DataFrame(differences)


def plot_effect_heatmap(effects: pd.DataFrame, output: Path) -> None:
    local = effects[effects.kind == "abs_db"]
    matrix = local.pivot(index="channel", columns="band", values="cohens_dz")
    matrix = matrix.reindex(columns=list(BANDS))
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values(ascending=False).index]
    limit = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(7.2, 9), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(
        xticks=np.arange(len(matrix.columns)),
        xticklabels=matrix.columns,
        yticks=np.arange(len(matrix.index)),
        yticklabels=matrix.index,
        xlabel="Frequency band",
        ylabel="EEG channel",
        title="Paired physiological effect: Fatigue1+2 minus Wakefulness",
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Cohen's dz of absolute band power (dB)")
    save_figure(fig, output, "physiology_channel_band_effect_heatmap")


def plot_effect_topomaps(effects: pd.DataFrame, output: Path) -> None:
    local = effects[effects.kind == "abs_db"]
    channels = sorted(local.channel.unique())
    info = mne.create_info(channels, sfreq=200.0, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="raise")
    values = {
        band: (
            local[local.band == band]
            .set_index("channel")
            .reindex(channels)["cohens_dz"]
            .to_numpy()
        )
        for band in BANDS
    }
    limit = max(
        0.5,
        max(float(np.nanmax(np.abs(array))) for array in values.values()),
    )
    fig, axes = plt.subplots(1, len(BANDS), figsize=(13, 3.4), constrained_layout=True)
    image = None
    for axis, (band, array) in zip(axes, values.items()):
        image, _ = mne.viz.plot_topomap(
            array,
            info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(-limit, limit),
            contours=6,
            sensors=True,
        )
        axis.set_title(f"{band.capitalize()} {BANDS[band][0]:g}-{BANDS[band][1]:g} Hz")
    colorbar = fig.colorbar(image, ax=axes)
    colorbar.set_label("Cohen's dz: Fatigue1+2 minus Wakefulness")
    fig.suptitle("Topographic effect sizes of absolute EEG band power")
    save_figure(fig, output, "physiology_bandpower_effect_topomaps")


def plot_progression(
    summary: pd.DataFrame,
    parsed: list[tuple[str, str, str] | None],
    output: Path,
    seed: int,
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    for feature_index, feature in enumerate(parsed):
        if feature is None:
            continue
        channel, kind, band = feature
        if kind != "rel":
            continue
        local = summary[summary.feature_index == feature_index]
        rows.append(
            local.assign(
                channel=channel,
                region=region_for(channel),
                band=band,
            )
        )
    long = pd.concat(rows, ignore_index=True)
    regional = (
        long.groupby(["subject", "label", "band", "region"], as_index=False)
        .agg(value=("value", "mean"), n_windows=("n_windows", "max"))
    )
    complete = (
        regional.groupby(["subject", "band", "region"])["label"]
        .nunique()
        .eq(3)
        .rename("complete_case")
        .reset_index()
    )
    regional = regional.merge(
        complete,
        on=["subject", "band", "region"],
        how="left",
    )
    regional = regional[regional.complete_case].copy()
    wake = (
        regional[regional.label == 0]
        .set_index(["subject", "band", "region"])["value"]
        .rename("wake")
    )
    regional = regional.join(wake, on=["subject", "band", "region"])
    regional["change_from_wake"] = regional["value"] - regional["wake"]
    regional = regional[np.isfinite(regional.change_from_wake)].copy()
    rng = np.random.default_rng(seed)
    aggregate_rows = []
    for keys, group in regional.groupby(["label", "band", "region"]):
        low, high = bootstrap_ci(
            group.change_from_wake.to_numpy(),
            rng,
            repetitions,
        )
        aggregate_rows.append(
            {
                "label": keys[0],
                "band": keys[1],
                "region": keys[2],
                "mean_change": group.change_from_wake.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "n_subjects": int(
                    group.loc[
                        np.isfinite(group.change_from_wake),
                        "subject",
                    ].nunique()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    labels = np.asarray([0, 1, 2])
    for axis, band in zip(axes.flat, BANDS):
        local = aggregate[aggregate.band == band]
        for region in REGIONS:
            points = local[local.region == region].set_index("label").reindex(labels)
            axis.plot(labels, points.mean_change, marker="o", label=region)
            axis.fill_between(
                labels,
                points.ci95_low.to_numpy(dtype=float),
                points.ci95_high.to_numpy(dtype=float),
                alpha=0.12,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            xticks=labels,
            xticklabels=[LABEL_NAMES[value] for value in labels],
            ylabel="Change in relative band power",
            title=f"{band.capitalize()} ({BANDS[band][0]:g}-{BANDS[band][1]:g} Hz)",
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Subject-centered EEG progression from wakefulness to Fatigue2")
    save_figure(fig, output, "physiology_regional_bandpower_progression")
    return regional


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary, names, parsed = load_subject_summaries(args.feature_dir)
    effects, differences = paired_effects(
        summary,
        names,
        parsed,
        args.seed,
        args.bootstrap_repetitions,
    )
    effects.to_csv(tables / "physiology_paired_effects.csv", index=False)
    differences.to_csv(
        tables / "physiology_subject_paired_differences.csv",
        index=False,
    )
    summary.to_csv(tables / "physiology_subject_label_feature_means.csv", index=False)
    plot_effect_heatmap(effects, args.output_dir)
    plot_effect_topomaps(effects, args.output_dir)
    regional = plot_progression(
        summary,
        parsed,
        args.output_dir,
        args.seed,
        args.bootstrap_repetitions,
    )
    regional.to_csv(
        tables / "physiology_regional_label_progression.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "figure": "physiology_channel_band_effect_heatmap.png",
                "caption": (
                    "Paired subject-level effect sizes for absolute band power; "
                    "positive values indicate greater power in Fatigue1+2."
                ),
            },
            {
                "figure": "physiology_bandpower_effect_topomaps.png",
                "caption": (
                    "Scalp distribution of paired absolute-band-power effect "
                    "sizes for Fatigue1+2 versus Wakefulness."
                ),
            },
            {
                "figure": "physiology_regional_bandpower_progression.png",
                "caption": (
                    "Subject-centered relative-power progression across the "
                    "three physician labels, with participant-bootstrap 95% CIs."
                ),
            },
        ]
    ).to_csv(tables / "physiology_figure_manifest.csv", index=False)
    print(f"[physiology] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
