#!/usr/bin/env bash
# Five-subject end-to-end smoke for the final Task A pipeline.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
CACHE_ROOT=data/derived/final_task_a_smoke
EXPERIMENT_ROOT=experiments/final_task_a_smoke
SUBJECTS=(01 02 03 04 05)

test -d "${RAW_ROOT}"
test ! -e "${EXPERIMENT_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
pytest -p no:cacheprovider tests -q

for scope in within loso; do
  filter_scope=subject_continuous
  if [ "${scope}" = within ]; then filter_scope=group_bounded; fi
  python scripts/extract_psd_features.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_${scope}" \
    --task A --preprocessing psd_classification --montage edf32 \
    --window-sec 1 --stride-sec 1 --filter-scope "${filter_scope}" \
    --subjects "${SUBJECTS[@]}"
  python scripts/extract_eeg_windows.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_${scope}" \
    --task A --preprocessing physiological_validation --montage edf32 \
    --window-sec 1 --stride-sec 1 --filter-scope "${filter_scope}" \
    --subjects "${SUBJECTS[@]}"
done

for protocol in within_subject loso; do
  scope=loso
  if [ "${protocol}" = within_subject ]; then scope=within; fi
  for model in psd_svm random_forest; do
    python scripts/run_nested_task_a.py \
      --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
      --task A \
      --feature-dir "${CACHE_ROOT}/psd_${scope}" \
      --output-root "${EXPERIMENT_ROOT}" --model "${model}" \
      --protocol "${protocol}" --max-folds 1
  done
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_${scope}" \
    --output-dir "${EXPERIMENT_ROOT}/${protocol}/eegnet" \
    --task A --protocol "${protocol}" --device cuda --selection-epochs 1 \
    --epochs 2 --patience 1 --max-folds 1 --seed 42
done

for protocol in within_subject loso; do
  for model in psd_svm random_forest eegnet; do
    test -s "${EXPERIMENT_ROOT}/${protocol}/${model}/metrics.json"
  done
  test -s "${EXPERIMENT_ROOT}/${protocol}/eegnet/checkpoints/fold_000.pt"
done
rm -rf "${CACHE_ROOT}"

echo "FINAL_TASK_A_SMOKE_COMPLETE"
