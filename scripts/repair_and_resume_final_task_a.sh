#!/usr/bin/env bash
# Archive the audited preliminary run, repair all within models, then resume LOSO.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
CACHE_ROOT=data/derived/final_task_a
EXPERIMENT_ROOT=experiments/final_task_a
RESULT_ROOT=results/final_task_a
ALLOWLIST=data/metadata/eligible_within_task_a.txt
SUPERSEDED="${EXPERIMENT_ROOT}/audit_superseded_8179bdf"
STATE_ROOT="${EXPERIMENT_ROOT}/.repair_state"
MANIFEST_ROOT="${EXPERIMENT_ROOT}/cache_manifests"

test -d "${RAW_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

mkdir -p "${STATE_ROOT}"
mkdir -p "${MANIFEST_ROOT}"
if [ ! -e "${STATE_ROOT}/preliminary_archived" ]; then
  mkdir -p "${SUPERSEDED}/within_subject"
  for model in psd_svm random_forest eegnet; do
    source_path="${EXPERIMENT_ROOT}/within_subject/${model}"
    destination="${SUPERSEDED}/within_subject/${model}"
    if [ -d "${source_path}" ]; then
      test ! -e "${destination}"
      mv "${source_path}" "${destination}"
    fi
  done
  touch "${STATE_ROOT}/preliminary_archived"
fi

echo "[Repair 1/4] regenerate within features and all corrected models"
if [ ! -s "${CACHE_ROOT}/psd_edf32_within/manifest.json" ]; then
  rm -rf "${CACHE_ROOT}/psd_edf32_within"
  python scripts/extract_psd_features.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_edf32_within" \
    --task A --preprocessing psd_classification --montage edf32 \
    --window-sec 1 --stride-sec 1 --filter-scope group_bounded
fi
if [ ! -s "${CACHE_ROOT}/eeg_edf32_within/manifest.json" ]; then
  rm -rf "${CACHE_ROOT}/eeg_edf32_within"
  python scripts/extract_eeg_windows.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_edf32_within" \
    --task A --preprocessing physiological_validation --montage edf32 \
    --window-sec 1 --stride-sec 1 --filter-scope group_bounded
fi
cp "${CACHE_ROOT}/psd_edf32_within/manifest.json" \
  "${MANIFEST_ROOT}/psd_edf32_within.json"
cp "${CACHE_ROOT}/eeg_edf32_within/manifest.json" \
  "${MANIFEST_ROOT}/eeg_edf32_within.json"
for model in psd_svm random_forest; do
  if [ ! -s "${EXPERIMENT_ROOT}/within_subject/${model}/metrics.json" ]; then
    find "${EXPERIMENT_ROOT}/within_subject" -maxdepth 1 -type d \
      -name ".${model}.partial-*" -exec rm -rf {} +
    python scripts/run_nested_task_a.py \
      --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
      --feature-dir "${CACHE_ROOT}/psd_edf32_within" \
      --output-root "${EXPERIMENT_ROOT}" --model "${model}" \
      --protocol within_subject --subject-allowlist "${ALLOWLIST}"
  fi
done
if [ ! -s "${EXPERIMENT_ROOT}/within_subject/eegnet/metrics.json" ]; then
  find "${EXPERIMENT_ROOT}/within_subject" -maxdepth 1 -type d \
    -name '.eegnet.partial-*' -exec rm -rf {} +
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_edf32_within" \
    --output-dir "${EXPERIMENT_ROOT}/within_subject/eegnet" \
    --protocol within_subject --subject-allowlist "${ALLOWLIST}" \
    --device cuda --seed 42
fi
rm -rf "${CACHE_ROOT}/psd_edf32_within" "${CACHE_ROOT}/eeg_edf32_within"

echo "[Repair 2/4] regenerate dB PSD LOSO and complete primary evaluation"
if [ ! -s "${CACHE_ROOT}/psd_edf32_loso/manifest.json" ]; then
  rm -rf "${CACHE_ROOT}/psd_edf32_loso"
  python scripts/extract_psd_features.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_edf32_loso" \
    --task A --preprocessing psd_classification --montage edf32 \
    --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
fi
cp "${CACHE_ROOT}/psd_edf32_loso/manifest.json" \
  "${MANIFEST_ROOT}/psd_edf32_loso.json"
for model in psd_svm random_forest; do
  if [ ! -s "${EXPERIMENT_ROOT}/loso/${model}/metrics.json" ]; then
    find "${EXPERIMENT_ROOT}/loso" -maxdepth 1 -type d \
      -name ".${model}.partial-*" -exec rm -rf {} +
    python scripts/run_nested_task_a.py \
      --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
      --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
      --output-root "${EXPERIMENT_ROOT}" --model "${model}" --protocol loso
  fi
done
if [ ! -s "${EXPERIMENT_ROOT}/loso/eegnet/metrics.json" ]; then
  if [ ! -s "${CACHE_ROOT}/eeg_edf32_loso/manifest.json" ]; then
    rm -rf "${CACHE_ROOT}/eeg_edf32_loso"
    python scripts/extract_eeg_windows.py \
      --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_edf32_loso" \
      --task A --preprocessing physiological_validation --montage edf32 \
      --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
  fi
  cp "${CACHE_ROOT}/eeg_edf32_loso/manifest.json" \
    "${MANIFEST_ROOT}/eeg_edf32_loso.json"
  find "${EXPERIMENT_ROOT}/loso" -maxdepth 1 -type d \
    -name '.eegnet.partial-*' -exec rm -rf {} +
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_edf32_loso" \
    --output-dir "${EXPERIMENT_ROOT}/loso/eegnet" \
    --protocol loso --device cuda --seed 42
fi
rm -rf "${CACHE_ROOT}/eeg_edf32_loso"

echo "[Repair 3/4] LOSO paper28 channel ablation"
if [ ! -s "${CACHE_ROOT}/psd_paper28_loso/manifest.json" ]; then
  rm -rf "${CACHE_ROOT}/psd_paper28_loso"
  python scripts/extract_psd_features.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/psd_paper28_loso" \
    --task A --preprocessing psd_classification --montage paper28 \
    --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
fi
if [ ! -s "${CACHE_ROOT}/eeg_paper28_loso/manifest.json" ]; then
  rm -rf "${CACHE_ROOT}/eeg_paper28_loso"
  python scripts/extract_eeg_windows.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/eeg_paper28_loso" \
    --task A --preprocessing physiological_validation --montage paper28 \
    --window-sec 1 --stride-sec 1 --filter-scope subject_continuous
fi
cp "${CACHE_ROOT}/psd_paper28_loso/manifest.json" \
  "${MANIFEST_ROOT}/psd_paper28_loso.json"
cp "${CACHE_ROOT}/eeg_paper28_loso/manifest.json" \
  "${MANIFEST_ROOT}/eeg_paper28_loso.json"
if [ ! -s "${EXPERIMENT_ROOT}/channel_paper28/loso/psd_svm/metrics.json" ]; then
  mkdir -p "${EXPERIMENT_ROOT}/channel_paper28/loso"
  find "${EXPERIMENT_ROOT}/channel_paper28/loso" -maxdepth 1 -type d \
    -name '.psd_svm.partial-*' -exec rm -rf {} +
  python scripts/run_nested_task_a.py \
    --raw-root "${RAW_ROOT}" --config configs/final/task_a_final.yaml \
    --feature-dir "${CACHE_ROOT}/psd_paper28_loso" \
    --output-root "${EXPERIMENT_ROOT}/channel_paper28" \
    --model psd_svm --protocol loso
fi
if [ ! -s "${EXPERIMENT_ROOT}/channel_paper28/loso/eegnet/metrics.json" ]; then
  find "${EXPERIMENT_ROOT}/channel_paper28/loso" -maxdepth 1 -type d \
    -name '.eegnet.partial-*' -exec rm -rf {} +
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_paper28_loso" \
    --output-dir "${EXPERIMENT_ROOT}/channel_paper28/loso/eegnet" \
    --protocol loso --device cuda --seed 42
fi
rm -rf "${CACHE_ROOT}/psd_paper28_loso" "${CACHE_ROOT}/eeg_paper28_loso"

echo "[Repair 4/4] publication tables and figures"
if [ -e "${RESULT_ROOT}" ] && [ ! -s "${RESULT_ROOT}/README.md" ]; then
  rm -rf "${RESULT_ROOT}"
fi
if [ ! -s "${RESULT_ROOT}/README.md" ]; then
  python scripts/assemble_final_task_a_results.py \
    --experiments-root "${EXPERIMENT_ROOT}" --output-dir "${RESULT_ROOT}" \
    --channel-ablation-root "${EXPERIMENT_ROOT}/channel_paper28"
fi
if [ ! -s "${RESULT_ROOT}/figures/tsne_psd_exploratory.png" ]; then
  python scripts/plot_tsne_task_a.py \
    --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
    --output-dir "${RESULT_ROOT}/figures" --seed 42
fi
rm -rf "${CACHE_ROOT}/psd_edf32_loso"

echo "FINAL_TASK_A_COMPLETE"
