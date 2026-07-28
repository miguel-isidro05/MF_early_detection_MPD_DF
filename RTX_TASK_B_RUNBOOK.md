# RTX Runbook: Task B

Task B compares Wakefulness against Fatigue1+Fatigue2. Run the five-subject
screen before the full benchmark. It checks execution and reports
preprocessing sensitivity; it neither selects nor changes the fixed primary
pipeline.

## 1. Update and verify

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
```

Continue only if every test passes.

## 2. Five-subject preprocessing screen

```bash
cd ~/FatigaMental
rm -rf \
  data/derived/task_b_hypothesis_smoke \
  experiments/task_b_hypothesis_smoke \
  results/task_b_hypothesis_smoke
mkdir -p logs

nohup bash scripts/run_task_b_hypothesis_smoke.sh \
  > logs/task_b_hypothesis_smoke.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f logs/task_b_hypothesis_smoke.log
```

The required final line is:

```text
TASK_B_HYPOTHESIS_SMOKE_COMPLETE
```

This run compares:

- 1-100 Hz and 0.3-35 Hz filtering;
- per-channel z-score, global z-score, and microvolt scaling;
- 1, 5, and 10-second windows;
- inclusion versus a 10-second guard around label transitions;
- PSD-SVM and fixed EEGNet diagnostics.

Package the screen for review:

```bash
cd ~/FatigaMental
test "$(tail -n 1 logs/task_b_hypothesis_smoke.log)" = \
  "TASK_B_HYPOTHESIS_SMOKE_COMPLETE"
tar -czf task_b_hypothesis_review.tar.gz \
  --exclude='*/checkpoints' \
  experiments/task_b_hypothesis_smoke \
  results/task_b_hypothesis_smoke \
  logs/task_b_hypothesis_smoke.log \
  FINAL_TASK_B_PROTOCOL.md \
  configs/final/task_b_final.yaml
ls -lh task_b_hypothesis_review.tar.gz
```

Review this archive for execution failures or implausible physiology before
launching the full run. Do not choose the primary profile from these
outer-fold scores; the primary profile is already fixed in
`configs/final/task_b_final.yaml`.

## 3. Full Task B benchmark

Once the diagnostic has been reviewed:

```bash
cd ~/FatigaMental
rm -rf data/derived/final_task_b experiments/final_task_b results/final_task_b
mkdir -p logs

nohup bash scripts/run_final_task_b.sh \
  > logs/final_task_b.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f logs/final_task_b.log
```

The required final line is:

```text
FINAL_TASK_B_COMPLETE
```

If the full run stops, keep `experiments/final_task_b` and resume only missing
stages:

```bash
cd ~/FatigaMental
nohup bash scripts/repair_and_resume_final_task_b.sh \
  > logs/final_task_b_resume.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f logs/final_task_b_resume.log
```

The resume script validates each completed `metrics.json`, rebuilds only a
missing cache, and does not retrain a completed model.

## 4. Package the final evidence

```bash
cd ~/FatigaMental
test "$(tail -n 1 logs/final_task_b.log)" = "FINAL_TASK_B_COMPLETE"
tar -czf task_b_final_review_bundle.tar.gz \
  --exclude='*/checkpoints' \
  --exclude='*.partial-*' \
  experiments/final_task_b \
  results/final_task_b \
  logs/final_task_b.log \
  FINAL_TASK_B_PROTOCOL.md \
  configs/final/task_b_final.yaml \
  data/metadata/eligible_within_task_b.txt \
  data/metadata/excluded_within_task_b.csv
ls -lh task_b_final_review_bundle.tar.gz
```

The result bundle contains out-of-fold predictions, nested selections,
subject-level metrics, ROC/PR and calibration plots, confusion matrices,
learning curves, t-SNE source data, physiological effect-size topographies,
regional band-power progression, and the source tables used for every new
physiological figure.
