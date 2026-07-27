# MPD-DF EEG Framework

Code and reproduction artifacts for **A Subject-Independent EEG Framework for
Early-Stage Driving Fatigue Recognition**.

The modeling input is EEG only. PSG is used once to recover the official temporal
overlap; `data/metadata/alignment_manifest.csv` freezes that timing so the portable
dataset does not need PSG waveforms.

## Verified dataset

- 50 EEG EDF files and 50 physician annotation files.
- 32 EDF channels, all in one channel order, sampled at 500 Hz.
- 372,404 aligned seconds, with exact agreement against the public `DataAlign.py`
  timing semantics.
- No NaN or Inf values in a full scan of all EEG samples.
- The EDF physical-unit bytes are malformed. This project interprets the stored
  values as microvolts and applies `1e-6` before analysis.
- The published Table 7 is not internally reproducible from the public raw files.
  Exact differences are stored in
  `reproduction/tables/table07_full_comparison.csv`.

## Local setup

```bash
conda activate fatigue-eeg-benchmark
python -m pip install -e .
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
pytest -p no:cacheprovider tests -q
```

The project-wide seed is `42`.

## Reproduction checks

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

## Model paths

PSD baselines use `extract_psd_features.py` followed by `run_classical.py`.
Deep models use a memory-mapped cache:

```bash
python scripts/extract_eeg_windows.py \
  --raw-root "data/raw/mpd_df_raw_full/Raw Dataset" \
  --output-dir data/derived/task_c_eeg \
  --task C \
  --preprocessing mne_eeglab_like \
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

The final RTX protocol intentionally omits the local inferred `MSCNNCAM` and
Tasks B/C. It runs nested PSD-SVM, Random Forest, and EEGNet for Task A only,
with within-subject grouped development, primary LOSO evaluation, and an
`edf32` versus `paper28` channel ablation. Run the short final-path smoke before
the full benchmark:

```bash
bash scripts/run_final_task_a_smoke.sh
bash scripts/run_final_task_a.sh
```

The full command creates `experiments/final_task_a` with out-of-fold
predictions, selections, and EEGNet checkpoints, then creates
`results/final_task_a` with publication tables and figures.
The public paper and repository omit the Table 9 training code, split,
classification preprocessing, metric averaging, channel subset, and artifact
policy, so no local model is reported as an exact Table 9 reproduction.

See `PREPROCESSING_PROTOCOL.md` for the paper-confirmed visualization profiles,
the standardized Task A/B benchmark profile, and the limits of Figure 11.

## RTX transfer

The portable data archive is written to `packages/mpd_df_eeg_only_rtx.zip`. It
contains EEG, annotations, and the frozen timing manifest. See
`environment/RTX4060_SETUP.md`, `RTX_EXECUTION_PLAN.md`, and
`RTX_TERMINAL_RUNBOOK.md` for the copy-paste execution sequence.

Archive size: 8,585,629,587 bytes. SHA-256:
`347f6dc9dd9926e64b43be0ac3c3e0d81215ae28e7619d1566e94d7035de3b09`.
