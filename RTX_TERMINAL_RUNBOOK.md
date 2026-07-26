# RTX Terminal Runbook

This branch runs only Task A with three models: PSD-SVM, Random Forest, and
EEGNet. It has two intentionally separate commands: a five-subject smoke run and
a 50-subject full benchmark.

## One-time setup

From the project root, after copying `mpd_df_eeg_only_rtx.zip` there:

```bash
conda activate eeg-diffusion
python -m pip install -e . --no-deps
export MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpddf_mpl PYTHONUNBUFFERED=1

unzip mpd_df_eeg_only_rtx.zip -d data/raw
rm mpd_df_eeg_only_rtx.zip

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
pytest -p no:cacheprovider tests -q
python scripts/verify_official_alignment.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output reproduction/reports/r0_alignment_rtx.json
```

The alignment verification must end with `"status": "MATCH"`.

## Command 1: five-subject smoke test

This runs only subjects `01-05`, keeps its six result folders, deletes only its
temporary caches, and then terminates. Send the contents of
`experiments/smoke/**/metrics.json` and `device.json` for review before running
the next command.

```bash
bash scripts/run_rtx_smoke_task_a.sh
```

Expected final line: `SMOKE_TASK_A_COMPLETE`.

## Command 2: full 50-subject Task A benchmark

Run this only after the smoke run has been reviewed. It refuses to start unless
the six smoke `metrics.json` files exist. It runs all 50 subjects, retains all
experiment outputs, deletes only regenerated caches, and then terminates.

```bash
bash scripts/run_rtx_full_task_a.sh
```

Expected final line: `FULL_TASK_A_COMPLETE`.

The production preprocessing is `mne_eeglab_like`: MNE zero-phase FIR,
0.3-35 Hz bandpass, 49-51 Hz notch, native 500 Hz, no z-score and no resampling.
All scripts use global seed 42.
