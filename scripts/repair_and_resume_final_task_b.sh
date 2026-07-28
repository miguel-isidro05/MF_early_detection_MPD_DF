#!/usr/bin/env bash
# Resume Task B without repeating completed model evaluations.
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
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p "${MANIFEST_ROOT}"

ensure_psd_cache() {
  local name="$1"
  local montage="$2"
  local scope="$3"
  if [ ! -s "${CACHE_ROOT}/${name}/manifest.json" ]; then
    mkdir -p "${CACHE_ROOT}"
    python scripts/extract_psd_features.py \
      --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/${name}" \
      --task B --preprocessing psd_classification --montage "${montage}" \
      --window-sec 1 --stride-sec 1 --filter-scope "${scope}"
  fi
  cp "${CACHE_ROOT}/${name}/manifest.json" "${MANIFEST_ROOT}/${name}.json"
}

ensure_eeg_cache() {
  local name="$1"
  local montage="$2"
  local scope="$3"
  if [ ! -s "${CACHE_ROOT}/${name}/manifest.json" ]; then
    mkdir -p "${CACHE_ROOT}"
    python scripts/extract_eeg_windows.py \
      --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/${name}" \
      --task B --preprocessing physiological_validation --montage "${montage}" \
      --window-sec 1 --stride-sec 1 --filter-scope "${scope}"
  fi
  cp "${CACHE_ROOT}/${name}/manifest.json" "${MANIFEST_ROOT}/${name}.json"
}

echo "[Resume 1/4] Task B within-subject"
if [ ! -s "${EXPERIMENT_ROOT}/within_subject/psd_svm/metrics.json" ] || \
   [ ! -s "${EXPERIMENT_ROOT}/within_subject/random_forest/metrics.json" ] || \
   [ ! -s "${RESULT_ROOT}/figures/physiology_bandpower_effect_topomaps.png" ]; then
  ensure_psd_cache psd_edf32_within edf32 group_bounded
fi
for model in psd_svm random_forest; do
  if [ ! -s "${EXPERIMENT_ROOT}/within_subject/${model}/metrics.json" ]; then
    python scripts/run_nested_task_a.py \
      --raw-root "${RAW_ROOT}" --config "${CONFIG}" --task B \
      --feature-dir "${CACHE_ROOT}/psd_edf32_within" \
      --output-root "${EXPERIMENT_ROOT}" --model "${model}" \
      --protocol within_subject --subject-allowlist "${ALLOWLIST}"
  fi
done
if [ ! -s "${EXPERIMENT_ROOT}/within_subject/eegnet/metrics.json" ]; then
  ensure_eeg_cache eeg_edf32_within edf32 group_bounded
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_edf32_within" \
    --output-dir "${EXPERIMENT_ROOT}/within_subject/eegnet" \
    --task B --protocol within_subject --subject-allowlist "${ALLOWLIST}" \
    --device cuda --seed 42
fi
rm -rf "${CACHE_ROOT}/eeg_edf32_within"

echo "[Resume 2/4] Task B LOSO"
if [ ! -s "${EXPERIMENT_ROOT}/loso/psd_svm/metrics.json" ] || \
   [ ! -s "${EXPERIMENT_ROOT}/loso/random_forest/metrics.json" ] || \
   [ ! -s "${RESULT_ROOT}/figures/tsne_psd_exploratory.png" ]; then
  ensure_psd_cache psd_edf32_loso edf32 subject_continuous
fi
for model in psd_svm random_forest; do
  if [ ! -s "${EXPERIMENT_ROOT}/loso/${model}/metrics.json" ]; then
    python scripts/run_nested_task_a.py \
      --raw-root "${RAW_ROOT}" --config "${CONFIG}" --task B \
      --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
      --output-root "${EXPERIMENT_ROOT}" --model "${model}" --protocol loso
  fi
done
if [ ! -s "${EXPERIMENT_ROOT}/loso/eegnet/metrics.json" ]; then
  ensure_eeg_cache eeg_edf32_loso edf32 subject_continuous
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_edf32_loso" \
    --output-dir "${EXPERIMENT_ROOT}/loso/eegnet" \
    --task B --protocol loso --device cuda --seed 42
fi
rm -rf "${CACHE_ROOT}/eeg_edf32_loso"

echo "[Resume 3/4] Task B paper28 sensitivity"
if [ ! -s "${EXPERIMENT_ROOT}/channel_paper28/loso/psd_svm/metrics.json" ]; then
  ensure_psd_cache psd_paper28_loso paper28 subject_continuous
  python scripts/run_nested_task_a.py \
    --raw-root "${RAW_ROOT}" --config "${CONFIG}" --task B \
    --feature-dir "${CACHE_ROOT}/psd_paper28_loso" \
    --output-root "${EXPERIMENT_ROOT}/channel_paper28" \
    --model psd_svm --protocol loso
fi
if [ ! -s "${EXPERIMENT_ROOT}/channel_paper28/loso/eegnet/metrics.json" ]; then
  ensure_eeg_cache eeg_paper28_loso paper28 subject_continuous
  python scripts/run_nested_deep.py \
    --cache-dir "${CACHE_ROOT}/eeg_paper28_loso" \
    --output-dir "${EXPERIMENT_ROOT}/channel_paper28/loso/eegnet" \
    --task B --protocol loso --device cuda --seed 42
fi
rm -rf "${CACHE_ROOT}/psd_paper28_loso" "${CACHE_ROOT}/eeg_paper28_loso"

echo "[Resume 4/4] Task B result bundle"
if [ ! -s "${RESULT_ROOT}/README.md" ]; then
  python scripts/assemble_final_task_a_results.py \
    --experiments-root "${EXPERIMENT_ROOT}" --output-dir "${RESULT_ROOT}" \
    --task B --protocol-config "${CONFIG}" \
    --channel-ablation-root "${EXPERIMENT_ROOT}/channel_paper28"
fi
if [ ! -s "${RESULT_ROOT}/figures/tsne_psd_exploratory.png" ]; then
  ensure_psd_cache psd_edf32_loso edf32 subject_continuous
  python scripts/plot_tsne_task_a.py \
    --feature-dir "${CACHE_ROOT}/psd_edf32_loso" \
    --output-dir "${RESULT_ROOT}/figures" --task B --seed 42
fi
if [ ! -s "${RESULT_ROOT}/figures/physiology_bandpower_effect_topomaps.png" ]; then
  ensure_psd_cache psd_edf32_within edf32 group_bounded
  python scripts/plot_task_b_physiology.py \
    --feature-dir "${CACHE_ROOT}/psd_edf32_within" \
    --output-dir "${RESULT_ROOT}/figures" --seed 42
fi
if [ -s results/final_task_a/tables/model_summary.csv ] && \
   [ ! -s "${RESULT_ROOT}/figures/task_a_b_comparison.png" ]; then
  python scripts/plot_task_a_b_comparison.py \
    --task-a-results results/final_task_a \
    --task-b-results "${RESULT_ROOT}" \
    --output-dir "${RESULT_ROOT}/figures"
fi
rm -rf "${CACHE_ROOT}"

echo "FINAL_TASK_B_COMPLETE"
