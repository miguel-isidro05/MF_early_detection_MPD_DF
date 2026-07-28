#!/usr/bin/env bash
# Complete Task A study with staged cache cleanup.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
CACHE_ROOT=data/derived/final_task_a
EXPERIMENT_ROOT=experiments/final_task_a
RESULT_ROOT=results/final_task_a
ALLOWLIST=data/metadata/eligible_within_task_a.txt
MANIFEST_ROOT="${EXPERIMENT_ROOT}/cache_manifests"

test -d "${RAW_ROOT}"
test ! -e "${EXPERIMENT_ROOT}"
test ! -e "${RESULT_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p "${MANIFEST_ROOT}"

# Stage 1: personalized development on full 32-channel montage.
echo "[Stage 1/4] within-subject edf32"
python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_edf32_within" \
  --task A --preprocessing psd_classification --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded
python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_edf32_within" \
  --task A --preprocessing physiological_validation --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope group_bounded
cp "${CACHE_ROOT}/psd_edf32_within/manifest.json" \
  "${MANIFEST_ROOT}/psd_edf32_within.json"
cp "${CACHE_ROOT}/eeg_edf32_within/manifest.json" \
  "${MANIFEST_ROOT}/eeg_edf32_within.json"
for model in psd_svm random_forest; do
  python scripts/run_nested_task_a.py \
    --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
    --feature-dir "${CACHE_ROOT}/psd_edf32_within" \
    --output-root "${EXPERIMENT_ROOT}" --model "${model}" \
    --protocol within_subject --subject-allowlist "${ALLOWLIST}"
done
python scripts/run_nested_deep.py \
  --cache-dir "${CACHE_ROOT}/eeg_edf32_within" \
  --output-dir "${EXPERIMENT_ROOT}/within_subject/eegnet" \
  --protocol within_subject --subject-allowlist "${ALLOWLIST}" --device cuda --seed 42
rm -rf "${CACHE_ROOT}/psd_edf32_within" "${CACHE_ROOT}/eeg_edf32_within"

# Stage 2: primary unseen-subject LOSO on edf32.
echo "[Stage 2/4] LOSO edf32"
python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_edf32_loso" \
  --task A --preprocessing psd_classification --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_edf32_loso" \
  --task A --preprocessing physiological_validation --montage edf32 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
cp "${CACHE_ROOT}/psd_edf32_loso/manifest.json" \
  "${MANIFEST_ROOT}/psd_edf32_loso.json"
cp "${CACHE_ROOT}/eeg_edf32_loso/manifest.json" \
  "${MANIFEST_ROOT}/eeg_edf32_loso.json"
for model in psd_svm random_forest; do
  python scripts/run_nested_task_a.py \
    --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
    --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
    --output-root "${EXPERIMENT_ROOT}" --model "${model}" --protocol loso
done
python scripts/run_nested_deep.py \
  --cache-dir "${CACHE_ROOT}/eeg_edf32_loso" \
  --output-dir "${EXPERIMENT_ROOT}/loso/eegnet" \
  --protocol loso --device cuda --seed 42
rm -rf "${CACHE_ROOT}/eeg_edf32_loso"

# Stage 3: paper28 channel ablation under LOSO.
echo "[Stage 3/4] LOSO paper28 channel ablation"
python scripts/extract_psd_features.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_paper28_loso" \
  --task A --preprocessing psd_classification --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
python scripts/extract_eeg_windows.py \
  --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_paper28_loso" \
  --task A --preprocessing physiological_validation --montage paper28 \
  --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
cp "${CACHE_ROOT}/psd_paper28_loso/manifest.json" \
  "${MANIFEST_ROOT}/psd_paper28_loso.json"
cp "${CACHE_ROOT}/eeg_paper28_loso/manifest.json" \
  "${MANIFEST_ROOT}/eeg_paper28_loso.json"
python scripts/run_nested_task_a.py \
  --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
  --feature-dir "${CACHE_ROOT}/psd_paper28_loso" \
  --output-root "${EXPERIMENT_ROOT}/channel_paper28" --model psd_svm --protocol loso
python scripts/run_nested_deep.py \
  --cache-dir "${CACHE_ROOT}/eeg_paper28_loso" \
  --output-dir "${EXPERIMENT_ROOT}/channel_paper28/loso/eegnet" \
  --protocol loso --device cuda --seed 42
rm -rf "${CACHE_ROOT}/psd_paper28_loso" "${CACHE_ROOT}/eeg_paper28_loso"

echo "[Stage 4/4] publication tables and figures"
python scripts/assemble_final_task_a_results.py \
  --experiments-root "${EXPERIMENT_ROOT}" --output-dir "${RESULT_ROOT}" \
  --channel-ablation-root "${EXPERIMENT_ROOT}/channel_paper28"
python scripts/plot_tsne_task_a.py \
  --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
  --output-dir "${RESULT_ROOT}/figures" --seed 42
rm -rf "${CACHE_ROOT}/psd_edf32_loso"

echo "FINAL_TASK_A_COMPLETE"
