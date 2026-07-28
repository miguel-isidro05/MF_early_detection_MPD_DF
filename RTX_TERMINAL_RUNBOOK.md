# RTX Terminal Runbook: Final Task A

This branch executes only Task A (Wakefulness versus Fatigue1). The final
models are nested PSD-SVM, Random Forest, and EEGNet. MSCNN-CAM and Tasks B/C
are excluded by study decision. The global seed is 42.

## 1. Update and verify

Run from the SSH home directory:

```bash
cd ~/FatigaMental
git fetch origin
git checkout rtx-eeg-pipeline
git pull --ff-only origin rtx-eeg-pipeline

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg-diffusion
python -m pip install -e . --no-deps

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

test "$(find data/raw/MPD_DF_EEG_ONLY/EEG -name '*.edf' | wc -l)" -eq 50
test "$(find data/raw/MPD_DF_EEG_ONLY/Annotation -name '*.txt' | wc -l)" -eq 50
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
pytest -p no:cacheprovider tests -q
python scripts/verify_official_alignment.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output reproduction/reports/r0_alignment_rtx.json
```

Continue only if the tests pass and the alignment report ends in `MATCH`.

## 2. Five-subject final-path smoke

Remove only an earlier incomplete final smoke, then launch:

```bash
cd ~/FatigaMental
rm -rf data/derived/final_task_a_smoke experiments/final_task_a_smoke
mkdir -p logs

nohup bash scripts/run_final_task_a_smoke.sh \
  > logs/final_task_a_smoke.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f logs/final_task_a_smoke.log
```

The required last line is:

```text
FINAL_TASK_A_SMOKE_COMPLETE
```

The smoke runs the exact nested code path with one outer fold and shortened
EEGNet epochs. It is an execution check, not a paper result.

## 3. Full final benchmark

Only after the smoke completes:

```bash
cd ~/FatigaMental
test "$(tail -n 1 logs/final_task_a_smoke.log)" = "FINAL_TASK_A_SMOKE_COMPLETE"
rm -rf data/derived/final_task_a
mkdir -p logs

nohup bash scripts/run_final_task_a.sh \
  > logs/final_task_a.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f logs/final_task_a.log
```

The required last line is:

```text
FINAL_TASK_A_COMPLETE
```

Do not delete `experiments/final_task_a` or `results/final_task_a`: they contain
the out-of-fold predictions, fold metrics, nested selections, checkpoints,
tables, source data, and publication figures. Regenerable EEG/PSD caches are
deleted stage by stage.

## 4. Package results for review

```bash
cd ~/FatigaMental
test "$(tail -n 1 logs/final_task_a.log)" = "FINAL_TASK_A_COMPLETE"
tar -czf task_a_final_review_bundle.tar.gz \
  --exclude='*/checkpoints' \
  --exclude='*.partial-*' \
  experiments/final_task_a \
  results/final_task_a \
  logs/final_task_a.log \
  reproduction/reports/r0_alignment_rtx.json \
  audits/TASK_A_WITHIN_AUDIT_8179bdf.md \
  FINAL_TASK_A_PROTOCOL.md \
  configs/final/task_a_final.yaml \
  data/metadata/eligible_within_task_a.txt \
  data/metadata/excluded_within_task_a.csv
ls -lh task_a_final_review_bundle.tar.gz
```

Download `~/FatigaMental/task_a_final_review_bundle.tar.gz` for the final
scientific audit.

## Repair and resume after the audited stopped run

The first stopped run exposed non-converged SVM fits, non-logarithmic PSD
features, and an EEGNet refit budget based on the early-stopping history length
instead of the best validation epoch. Archive all preliminary within-subject
results, regenerate the features, repeat all three models, and then resume LOSO:

```bash
git pull --ff-only origin rtx-eeg-pipeline
nohup bash scripts/repair_and_resume_final_task_a.sh \
  > logs/final_task_a_repair_resume.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f logs/final_task_a_repair_resume.log
```

The repair command keeps all preliminary within-subject results under
`experiments/final_task_a/audit_superseded_8179bdf` and requires the complete
existing EEG LOSO cache.

When it finishes, package the corrected run without the large model
checkpoints:

```bash
test "$(tail -n 1 logs/final_task_a_repair_resume.log)" = "FINAL_TASK_A_COMPLETE"
tar -czf task_a_final_corrected_review_bundle.tar.gz \
  --exclude='*/checkpoints' \
  --exclude='*.partial-*' \
  experiments/final_task_a \
  results/final_task_a \
  logs/final_task_a_repair_resume.log \
  reproduction/reports/r0_alignment_rtx.json \
  audits/TASK_A_WITHIN_AUDIT_8179bdf.md \
  FINAL_TASK_A_PROTOCOL.md \
  configs/final/task_a_final.yaml \
  data/metadata/eligible_within_task_a.txt \
  data/metadata/excluded_within_task_a.csv
ls -lh task_a_final_corrected_review_bundle.tar.gz
```

## Fixed protocol

- 1-second non-overlapping windows contained inside physician annotation bins.
- EEG: 1-100 Hz zero-phase bandpass, 50 Hz notch, mean removal, 200 Hz.
- EEGNet: per-window/per-channel z-score.
- PSD models: physical-amplitude PSD/log features and training-fold scaler.
- Within-subject grouped development plus primary unseen-subject LOSO.
- Hyperparameters and the Fatigue1 threshold are selected only from grouped
  inner validation.
- Primary montage `edf32`; LOSO channel ablation `paper28`.
- ICA is excluded from classification and retained only for the manual
  topographic reproduction.
