# Commands

Run from the project root after `python -m pip install -e .`.

## Audit and paper data

```bash
python scripts/verify_official_alignment.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output reproduction/reports/r0_alignment_verification.json

python scripts/audit_dataset.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output-dir data/metadata \
  --signal-qc full

python scripts/reproduce_table7.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output reproduction/tables/table07_full_comparison.csv
```

## Portable EEG archive

```bash
python scripts/build_alignment_manifest.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output data/metadata/alignment_manifest.csv

python scripts/package_eeg_only.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --alignment-manifest data/metadata/alignment_manifest.csv \
  --output packages/mpd_df_eeg_only_rtx.zip
```

## Deep cache and training

```bash
python scripts/extract_eeg_windows.py \
  --raw-root MPD_DF_EEG_ONLY \
  --output-dir data/derived/task_c_eeg \
  --task C \
  --preprocessing physiological_validation \
  --montage paper28 \
  --filter-scope group_bounded

python scripts/run_deep.py \
  --cache-dir data/derived/task_c_eeg \
  --output-dir experiments/within_subject/task_c_eegnet \
  --project-root . \
  --task C \
  --model eegnet \
  --protocol within_subject \
  --device cuda
```
