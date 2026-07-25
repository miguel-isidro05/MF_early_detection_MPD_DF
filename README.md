# MPD-DF EEG Framework

Research workspace for **A Subject-Independent EEG Framework for Early-Stage Driving Fatigue Recognition**.

The modeling scope is EEG-only. The original dataset remains under `../Dataset/28455737`; symlinks and derived manifests avoid duplicating the full download. PSG samples are not model inputs. PSG headers are read only to reproduce the official temporal overlap, and EOG may later support a documented ICA sensitivity analysis.

## Verified Data

- 50 EEG EDF files and 50 annotation files.
- 32 EEG channels in a single consistent order.
- Sampling rate: 500 Hz.
- Official-alignment label expansion: 372,404 seconds.
- No NaN/Inf values in the full EEG scan.
- EDF unit bytes are malformed; the loader explicitly interprets stored values as microvolts and applies `1e-6` to obtain volts.

## Execution

The global seed is `42`.

```bash
MNE_DONTWRITE_HOME=true PYTHONPATH=src pytest tests -q

MNE_DONTWRITE_HOME=true PYTHONPATH=src python scripts/verify_official_alignment.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output reproduction/reports/r0_alignment_verification.json
```

Reference reproduction remains methodologically blocked by missing Table 9 split, preprocessing, metric-averaging, and exact MSCNN-CAM implementation details. Standardized experiments must remain labeled R2 and may not be described as the exact official pipeline.

