#!/usr/bin/env bash
# Complete Task B benchmark after the five-subject hypothesis screen is reviewed.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
CACHE_ROOT=data/derived/final_task_b
EXPERIMENT_ROOT=experiments/final_task_b
RESULT_ROOT=results/final_task_b
ALLOWLIST=data/metadata/eligible_within_task_b.txt
CONFIG=configs/final/task_b_final.yaml
MANIFEST_ROOT="${EXPERIMENT_ROOT}/cache_manifests"

test -d "${RAW_ROOT}"
test ! -e "${CACHE_ROOT}"
test ! -e "${EXPERIMENT_ROOT}"
test ! -e "${RESULT_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p "${MANIFEST_ROOT}"

echo "[Stage 1/4] Task B within-subject edf32"
python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_edf32_within" \
  --task B --preprocessing psd_classification --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded
python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_edf32_within" \
  --task B --preprocessing physiological_validation --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded
cp "${CACHE_ROOT}/psd_edf32_within/manifest.json" \
  "${MANIFEST_ROOT}/psd_edf32_within.json"
cp "${CACHE_ROOT}/eeg_edf32_within/manifest.json" \
  "${MANIFEST_ROOT}/eeg_edf32_within.json"
for model in psd_svm random_forest; do
  python scripts/run_nested_task_a.py \
    --raw-root "${RAW_ROOT}" --config "${CONFIG}" --task B \
    --feature-dir "${CACHE_ROOT}/psd_edf32_within" \
    --output-root "${EXPERIMENT_ROOT}" --model "${model}" \
    --protocol within_subject --subject-allowlist "${ALLOWLIST}"
done
python scripts/run_nested_deep.py \
  --cache-dir "${CACHE_ROOT}/eeg_edf32_within" \
  --output-dir "${EXPERIMENT_ROOT}/within_subject/eegnet" \
  --task B --protocol within_subject --subject-allowlist "${ALLOWLIST}" \
  --device cuda --seed 42
rm -rf "${CACHE_ROOT}/eeg_edf32_within"

echo "[Stage 2/4] Task B LOSO edf32"
python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_edf32_loso" \
  --task B --preprocessing psd_classification --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_edf32_loso" \
  --task B --preprocessing physiological_validation --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
cp "${CACHE_ROOT}/psd_edf32_loso/manifest.json" \
  "${MANIFEST_ROOT}/psd_edf32_loso.json"
cp "${CACHE_ROOT}/eeg_edf32_loso/manifest.json" \
  "${MANIFEST_ROOT}/eeg_edf32_loso.json"
for model in psd_svm random_forest; do
  python scripts/run_nested_task_a.py \
    --raw-root "${RAW_ROOT}" --config "${CONFIG}" --task B \
    --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
    --output-root "${EXPERIMENT_ROOT}" --model "${model}" --protocol loso
done
python scripts/run_nested_deep.py \
  --cache-dir "${CACHE_ROOT}/eeg_edf32_loso" \
  --output-dir "${EXPERIMENT_ROOT}/loso/eegnet" \
  --task B --protocol loso --device cuda --seed 42
rm -rf "${CACHE_ROOT}/eeg_edf32_loso"

echo "[Stage 3/4] Task B LOSO paper28 channel sensitivity"
python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_paper28_loso" \
  --task B --preprocessing psd_classification --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_paper28_loso" \
  --task B --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
cp "${CACHE_ROOT}/psd_paper28_loso/manifest.json" \
  "${MANIFEST_ROOT}/psd_paper28_loso.json"
cp "${CACHE_ROOT}/eeg_paper28_loso/manifest.json" \
  "${MANIFEST_ROOT}/eeg_paper28_loso.json"
python scripts/run_nested_task_a.py \
  --raw-root "${RAW_ROOT}" --config "${CONFIG}" --task B \
  --feature-dir "${CACHE_ROOT}/psd_paper28_loso" \
  --output-root "${EXPERIMENT_ROOT}/channel_paper28" \
  --model psd_svm --protocol loso
python scripts/run_nested_deep.py \
  --cache-dir "${CACHE_ROOT}/eeg_paper28_loso" \
  --output-dir "${EXPERIMENT_ROOT}/channel_paper28/loso/eegnet" \
  --task B --protocol loso --device cuda --seed 42
rm -rf "${CACHE_ROOT}/psd_paper28_loso" "${CACHE_ROOT}/eeg_paper28_loso"

echo "[Stage 4/4] Task B publication tables, predictive plots, and physiology"
python scripts/assemble_final_task_a_results.py \
  --experiments-root "${EXPERIMENT_ROOT}" --output-dir "${RESULT_ROOT}" \
  --task B --protocol-config "${CONFIG}" \
  --channel-ablation-root "${EXPERIMENT_ROOT}/channel_paper28"
python scripts/plot_tsne_task_a.py \
  --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
  --output-dir "${RESULT_ROOT}/figures" --task B --seed 42
python scripts/plot_task_b_physiology.py \
  --feature-dir "${CACHE_ROOT}/psd_edf32_within" \
  --output-dir "${RESULT_ROOT}/figures" --seed 42
if [ -s results/final_task_a/tables/model_summary.csv ]; then
  python scripts/plot_task_a_b_comparison.py \
    --task-a-results results/final_task_a \
    --task-b-results "${RESULT_ROOT}" \
    --output-dir "${RESULT_ROOT}/figures"
fi
rm -rf "${CACHE_ROOT}"

echo "FINAL_TASK_B_COMPLETE"
