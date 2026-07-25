#!/usr/bin/env python3
"""Create a Zip64 EEG-only MPD-DF bundle without staging another 11 GB tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from mpd_df.dataset import discover_subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression-level", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    subjects = discover_subjects(args.raw_root, include_psg=False)
    if len(subjects) != 50:
        raise ValueError(f"Expected 50 EEG subjects, found {len(subjects)}")
    manifest = {
        "dataset": "MPD-DF EEG-only portable bundle",
        "subjects": [files.subject for files in subjects],
        "contains_psg": False,
        "alignment_source": "PSG-derived timing frozen in alignment_manifest.csv",
        "eeg_unit_contract": "stored EDF values are interpreted as microvolts by this project",
    }
    readme = (
        "# MPD-DF EEG-only bundle\n\n"
        "Contains the 50 raw EEG EDF files, physician annotations, and a frozen "
        "alignment manifest derived from PSG headers. PSG waveforms are not included.\n\n"
        "Extract this ZIP and pass its root directory to the project scripts as "
        "`--raw-root`.\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.",
        suffix=".partial",
        dir=args.output.parent,
        delete=False,
    ) as partial_file:
        partial = Path(partial_file.name)
    try:
        with zipfile.ZipFile(
            partial,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=args.compression_level,
            allowZip64=True,
        ) as archive:
            archive.writestr("MPD_DF_EEG_ONLY/README.md", readme)
            archive.writestr(
                "MPD_DF_EEG_ONLY/bundle_manifest.json",
                json.dumps(manifest, indent=2) + "\n",
            )
            archive.write(
                args.alignment_manifest,
                "MPD_DF_EEG_ONLY/alignment_manifest.csv",
            )
            for files in subjects:
                archive.write(files.eeg, f"MPD_DF_EEG_ONLY/EEG/{files.eeg.name}")
                archive.write(
                    files.annotation,
                    f"MPD_DF_EEG_ONLY/Annotation/{files.annotation.name}",
                )
        partial.rename(args.output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
