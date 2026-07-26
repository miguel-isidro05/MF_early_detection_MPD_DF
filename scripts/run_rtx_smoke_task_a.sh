#!/usr/bin/env bash
# Five-subject technical smoke test. Run from the project root with eeg-diffusion active.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
TASK=A
SMOKE_SUBJECTS=configs/smoke_task_a_subjects.txt
SMOKE_IDS=$(tr '\n' ' ' < "${SMOKE_SUBJECTS}")

test -d "${RAW_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "data/derived/smoke_psd_${TASK}" \
  --task "${TASK}" --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded \
  --subjects ${SMOKE_IDS}

python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "data/derived/smoke_eeg_${TASK}" \
  --task "${TASK}" --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded \
  --subjects ${SMOKE_IDS}

for model in psd_svm random_forest; do
  for protocol in within_subject loso; do
    extra=()
    if [ "${protocol}" = "within_subject" ]; then
      extra=(--subject-allowlist "${SMOKE_SUBJECTS}")
    fi
    python scripts/run_classical.py \
      --feature-dir "data/derived/smoke_psd_${TASK}" \
      --output-dir "experiments/smoke/${protocol}/${TASK}_${model}" \
      --project-root . --task "${TASK}" --model "${model}" --protocol "${protocol}" \
      "${extra[@]}" --seed 42
  done
done

for protocol in within_subject loso; do
  extra=()
  if [ "${protocol}" = "within_subject" ]; then
    extra=(--subject-allowlist "${SMOKE_SUBJECTS}")
  fi
  python scripts/run_deep.py \
    --cache-dir "data/derived/smoke_eeg_${TASK}" \
    --output-dir "experiments/smoke/${protocol}/${TASK}_eegnet" \
    --project-root . --task "${TASK}" --model eegnet --protocol "${protocol}" \
    "${extra[@]}" --epochs 2 --patience 1 --batch-size 128 \
    --learning-rate 0.001 --weight-decay 0.0001 \
    --device cuda --num-workers 4 --seed 42 --max-folds 1
done

for path in \
  experiments/smoke/within_subject/A_psd_svm/metrics.json \
  experiments/smoke/within_subject/A_random_forest/metrics.json \
  experiments/smoke/within_subject/A_eegnet/metrics.json \
  experiments/smoke/loso/A_psd_svm/metrics.json \
  experiments/smoke/loso/A_random_forest/metrics.json \
  experiments/smoke/loso/A_eegnet/metrics.json; do
  test -f "${path}"
done

rm -rf data/derived/smoke_psd_A data/derived/smoke_eeg_A
echo "SMOKE_TASK_A_COMPLETE"
