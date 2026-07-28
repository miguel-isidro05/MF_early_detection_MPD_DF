#!/usr/bin/env python3
"""Exploratory t-SNE of PSD features; never used for model selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from run_classical import load_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=("A", "B"), default="A")
    args = parser.parse_args()
    X, y, metadata, _ = load_features(args.feature_dir, args.task)
    subject = metadata["subject"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(y), size=min(args.max_points, len(y)), replace=False)
    X = StandardScaler().fit_transform(X[selected])
    X = PCA(n_components=min(30, X.shape[1]), random_state=args.seed).fit_transform(X)
    embedding = TSNE(
        n_components=2, init="pca", learning_rate="auto", perplexity=40,
        random_state=args.seed,
    ).fit_transform(X)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.DataFrame({
        "x": embedding[:, 0], "y": embedding[:, 1], "label": y[selected],
        "subject": subject[selected],
    })
    source.to_csv(args.output_dir / "tsne_psd_source.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    positive_name = "Fatigue1" if args.task == "A" else "Fatigue1 + Fatigue2"
    for label, name, color in (
        (0, "Wakefulness", "#277da1"),
        (1, positive_name, "#f94144"),
    ):
        rows = source[source.label == label]
        axis.scatter(rows.x, rows.y, s=7, alpha=.45, label=name, color=color)
    axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Exploratory PSD embedding (not inferential)")
    axis.legend()
    figure.savefig(args.output_dir / "tsne_psd_exploratory.png", dpi=300)
    figure.savefig(args.output_dir / "tsne_psd_exploratory.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
