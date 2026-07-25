#!/usr/bin/env python3
"""Capture the active runtime used for an audit or experiment."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def main() -> None:
    args = parse_args()
    import torch

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "pandas", "sklearn", "mne", "torch", "braindecode")
        },
        "torch": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "note": (
            "This captures the local audit/development machine. Final RTX 4060 runs "
            "must capture their own NVIDIA driver, CUDA, cuDNN, and GPU details."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

