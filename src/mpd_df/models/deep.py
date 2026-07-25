"""Validated EEGNet and auditable local MSCNN-CAM reimplementation."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact EEGNet-style binary classifier for (batch, channels, time)."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = temporal_kernel // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, padding), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                (n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                (1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            output = self.features(torch.zeros(1, 1, n_channels, n_times))
        self.classifier = nn.Linear(output.numel(), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("EEGNet input must have shape (batch, channels, time)")
        features = self.features(x.unsqueeze(1))
        return self.classifier(features.flatten(start_dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class MSCNNCAM(nn.Module):
    """Local MSCNN-CAM-compatible interpretation.

    This is intentionally labeled a local reimplementation. Exact equivalence
    cannot be claimed until the cited 2024 implementation and its training
    protocol are available.
    """

    def __init__(
        self,
        n_channels: int,
        branch_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (7, 15, 31),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_channels, branch_filters, kernel, padding=kernel // 2),
                    nn.BatchNorm1d(branch_filters),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                )
                for kernel in kernel_sizes
            ]
        )
        merged = branch_filters * len(kernel_sizes)
        self.attention = ChannelAttention(merged)
        self.head = nn.Sequential(
            nn.Conv1d(merged, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.head(self.attention(features))
