# Commands

## Audit

```bash
MNE_DONTWRITE_HOME=true PYTHONPATH=src python scripts/audit_dataset.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output-dir data/metadata \
  --signal-qc full
```

## PSD cache

```bash
MNE_DONTWRITE_HOME=true PYTHONPATH=src python scripts/extract_psd_features.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output-dir data/derived/task_c_psd \
  --task C \
  --preprocessing physiological_validation
```

## Standardized LOSO baseline

```bash
PYTHONPATH=src python scripts/run_classical.py \
  --feature-dir data/derived/task_c_psd \
  --output-dir reproduction/runs/r2_psd_svm_loso \
  --project-root . \
  --task C \
  --model psd_svm \
  --protocol loso
```

