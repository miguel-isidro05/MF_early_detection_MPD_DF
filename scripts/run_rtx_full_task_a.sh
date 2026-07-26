#!/usr/bin/env bash
# Full 50-subject Task A benchmark. Launch only after reviewing the smoke results.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
TASK=A
ALLOWLIST=data/metadata/eligible_within_task_a.txt

for path in \
  experiments/smoke/within_subject/A_psd_svm/metrics.json \
  experiments/smoke/within_subject/A_random_forest/metrics.json \
  experiments/smoke/within_subject/A_eegnet/metrics.json \
  experiments/smoke/loso/A_psd_svm/metrics.json \
  experiments/smoke/loso/A_random_forest/metrics.json \
  experiments/smoke/loso/A_eegnet/metrics.json; do
  test -f "${path}"
done
test -d "${RAW_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "data/derived/psd_${TASK}_1s_paper28" \
  --task "${TASK}" --preprocessing mne_eeglab_like --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded

python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "data/derived/eeg_${TASK}_1s_paper28" \
  --task "${TASK}" --preprocessing mne_eeglab_like --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded

for model in psd_svm random_forest; do
  python scripts/run_classical.py \
    --feature-dir "data/derived/psd_${TASK}_1s_paper28" \
    --output-dir "experiments/within_subject/${TASK}_1s_${model}" \
    --project-root . --task "${TASK}" --model "${model}" --protocol within_subject \
    --subject-allowlist "${ALLOWLIST}" --seed 42
  python scripts/run_classical.py \
    --feature-dir "data/derived/psd_${TASK}_1s_paper28" \
    --output-dir "experiments/loso/${TASK}_1s_${model}" \
    --project-root . --task "${TASK}" --model "${model}" --protocol loso --seed 42
done

for protocol in within_subject loso; do
  extra=()
  if [ "${protocol}" = "within_subject" ]; then
    extra=(--subject-allowlist "${ALLOWLIST}")
  fi
  python scripts/run_deep.py \
    --cache-dir "data/derived/eeg_${TASK}_1s_paper28" \
    --output-dir "experiments/${protocol}/${TASK}_1s_eegnet" \
    --project-root . --task "${TASK}" --model eegnet --protocol "${protocol}" \
    "${extra[@]}" --epochs 50 --patience 8 --batch-size 128 \
    --learning-rate 0.001 --weight-decay 0.0001 \
    --device cuda --num-workers 4 --seed 42
done

for path in \
  experiments/within_subject/A_1s_psd_svm/metrics.json \
  experiments/within_subject/A_1s_random_forest/metrics.json \
  experiments/within_subject/A_1s_eegnet/metrics.json \
  experiments/loso/A_1s_psd_svm/metrics.json \
  experiments/loso/A_1s_random_forest/metrics.json \
  experiments/loso/A_1s_eegnet/metrics.json; do
  test -f "${path}"
done

rm -rf data/derived/psd_A_1s_paper28 data/derived/eeg_A_1s_paper28
echo "FULL_TASK_A_COMPLETE"
