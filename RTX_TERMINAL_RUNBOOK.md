# RTX Terminal Runbook

Run every command in `bash` from the project root. The project and
`mpd_df_eeg_only_rtx.zip` must be on the RTX machine.

## 1. Prepare the machine

```bash
unzip packages/mpd_df_eeg_only_rtx.zip -d data/raw
cd packages
sha256sum -c SHA256SUMS.txt
cd ..

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
python scripts/capture_environment.py --output environment/system_info_rtx4060.txt
python -m pip freeze > environment/requirements-lock-rtx4060.txt
pytest -p no:cacheprovider tests -q
```

Expected archive root: `data/raw/MPD_DF_EEG_ONLY`.

## 2. Verify the portable dataset

```bash
python scripts/verify_official_alignment.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output reproduction/reports/r0_alignment_rtx.json
```

The command must end with `"status": "MATCH"`. Do not train if it does not.

## 3. Run Task A, then free its caches

Task A is wakefulness versus early fatigue. `group_bounded` is used for both
protocols so that no filtered signal crosses an evaluation-group boundary.
The command does not delete results, only the regenerable caches after every A
experiment has written its `metrics.json`.

```bash
TASK=A
ALLOWLIST=data/metadata/eligible_within_task_a.txt

python scripts/extract_psd_features.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output-dir data/derived/psd_${TASK}_1s_paper28 \
  --task ${TASK} --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded

python scripts/extract_eeg_windows.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output-dir data/derived/eeg_${TASK}_1s_paper28 \
  --task ${TASK} --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded

for MODEL in psd_svm random_forest; do
  python scripts/run_classical.py \
    --feature-dir data/derived/psd_${TASK}_1s_paper28 \
    --output-dir experiments/within_subject/${TASK}_1s_${MODEL} \
    --project-root . --task ${TASK} --model ${MODEL} --protocol within_subject \
    --subject-allowlist ${ALLOWLIST} --seed 42
  python scripts/run_classical.py \
    --feature-dir data/derived/psd_${TASK}_1s_paper28 \
    --output-dir experiments/loso/${TASK}_1s_${MODEL} \
    --project-root . --task ${TASK} --model ${MODEL} --protocol loso --seed 42
done

for MODEL in eegnet mscnn_cam; do
  for PROTOCOL in within_subject loso; do
    EXTRA=()
    if [ "${PROTOCOL}" = "within_subject" ]; then EXTRA=(--subject-allowlist ${ALLOWLIST}); fi
    python scripts/run_deep.py \
      --cache-dir data/derived/eeg_${TASK}_1s_paper28 \
      --output-dir experiments/${PROTOCOL}/${TASK}_1s_${MODEL} \
      --project-root . --task ${TASK} --model ${MODEL} --protocol ${PROTOCOL} \
      "${EXTRA[@]}" --epochs 50 --patience 8 --batch-size 128 \
      --learning-rate 0.001 --weight-decay 0.0001 \
      --device cuda --num-workers 4 --seed 42
  done
done

test -f experiments/within_subject/A_1s_psd_svm/metrics.json
test -f experiments/within_subject/A_1s_random_forest/metrics.json
test -f experiments/within_subject/A_1s_eegnet/metrics.json
test -f experiments/within_subject/A_1s_mscnn_cam/metrics.json
test -f experiments/loso/A_1s_psd_svm/metrics.json
test -f experiments/loso/A_1s_random_forest/metrics.json
test -f experiments/loso/A_1s_eegnet/metrics.json
test -f experiments/loso/A_1s_mscnn_cam/metrics.json

rm -rf data/derived/psd_A_1s_paper28 data/derived/eeg_A_1s_paper28
```

## 4. Run Task B, then free its caches

Task B is wakefulness versus early and moderate fatigue. It follows the same
benchmark protocol and removes only its caches after all eight runs are present.

```bash
TASK=B
ALLOWLIST=data/metadata/eligible_within_task_b.txt

python scripts/extract_psd_features.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output-dir data/derived/psd_${TASK}_1s_paper28 \
  --task ${TASK} --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded

python scripts/extract_eeg_windows.py \
  --raw-root data/raw/MPD_DF_EEG_ONLY \
  --output-dir data/derived/eeg_${TASK}_1s_paper28 \
  --task ${TASK} --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded

for MODEL in psd_svm random_forest; do
  python scripts/run_classical.py \
    --feature-dir data/derived/psd_${TASK}_1s_paper28 \
    --output-dir experiments/within_subject/${TASK}_1s_${MODEL} \
    --project-root . --task ${TASK} --model ${MODEL} --protocol within_subject \
    --subject-allowlist ${ALLOWLIST} --seed 42
  python scripts/run_classical.py \
    --feature-dir data/derived/psd_${TASK}_1s_paper28 \
    --output-dir experiments/loso/${TASK}_1s_${MODEL} \
    --project-root . --task ${TASK} --model ${MODEL} --protocol loso --seed 42
done

for MODEL in eegnet mscnn_cam; do
  for PROTOCOL in within_subject loso; do
    EXTRA=()
    if [ "${PROTOCOL}" = "within_subject" ]; then EXTRA=(--subject-allowlist ${ALLOWLIST}); fi
    python scripts/run_deep.py \
      --cache-dir data/derived/eeg_${TASK}_1s_paper28 \
      --output-dir experiments/${PROTOCOL}/${TASK}_1s_${MODEL} \
      --project-root . --task ${TASK} --model ${MODEL} --protocol ${PROTOCOL} \
      "${EXTRA[@]}" --epochs 50 --patience 8 --batch-size 128 \
      --learning-rate 0.001 --weight-decay 0.0001 \
      --device cuda --num-workers 4 --seed 42
  done
done

test -f experiments/within_subject/B_1s_psd_svm/metrics.json
test -f experiments/within_subject/B_1s_random_forest/metrics.json
test -f experiments/within_subject/B_1s_eegnet/metrics.json
test -f experiments/within_subject/B_1s_mscnn_cam/metrics.json
test -f experiments/loso/B_1s_psd_svm/metrics.json
test -f experiments/loso/B_1s_random_forest/metrics.json
test -f experiments/loso/B_1s_eegnet/metrics.json
test -f experiments/loso/B_1s_mscnn_cam/metrics.json

rm -rf data/derived/psd_B_1s_paper28 data/derived/eeg_B_1s_paper28
```

## 5. Inspect results

```bash
find experiments -name metrics.json -print
find experiments -name excluded_subjects.csv -print
find experiments -name device.json -print
```

For each completed experiment, retain `metrics.json`, `subject_metrics.csv`,
`fold_metrics.csv`, `predictions.csv`, `split_assignments.csv`, `environment.txt`,
and, for deep models, checkpoints and learning curves. The primary outcome is the
`macro_subject.f1.mean` value in `metrics.json`, not an individual fold or the
window-pooled metric.
