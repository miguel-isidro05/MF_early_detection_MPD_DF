#!/usr/bin/env bash
# Five-subject causal screen for Task B preprocessing hypotheses.
set -euo pipefail

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/mpddf_mpl
export PYTHONUNBUFFERED=1

RAW_ROOT=data/raw/MPD_DF_EEG_ONLY
CACHE_ROOT=data/derived/task_b_hypothesis_smoke
EXPERIMENT_ROOT=experiments/task_b_hypothesis_smoke
RESULT_ROOT=results/task_b_hypothesis_smoke
ALLOWLIST=configs/smoke_task_a_subjects.txt
SUBJECTS=(01 02 03 04 05)

test -d "${RAW_ROOT}"
test ! -e "${CACHE_ROOT}"
test ! -e "${EXPERIMENT_ROOT}"
test ! -e "${RESULT_ROOT}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p "${CACHE_ROOT}" "${EXPERIMENT_ROOT}"

for profile in physiological_validation psd_classification annotation_classification; do
  for window in 1 5 10; do
    name="psd_${profile}_w${window}_g0"
    python scripts/extract_psd_features.py \
      --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/${name}" \
      --task B --preprocessing "${profile}" --montage edf32 \
      --window-sec "${window}" --stride-sec "${window}" \
      --filter-scope group_bounded --transition-guard-sec 0 \
      --subjects "${SUBJECTS[@]}"
    python scripts/run_classical.py \
      --feature-dir "${CACHE_ROOT}/${name}" \
      --output-dir "${EXPERIMENT_ROOT}/${name}" \
      --project-root . --task B --model psd_svm \
      --protocol within_subject --subject-allowlist "${ALLOWLIST}"
    if [ "${profile}" = annotation_classification ] && [ "${window}" = 5 ]; then
      mkdir -p "${RESULT_ROOT}/figures"
      python scripts/plot_task_b_physiology.py \
        --feature-dir "${CACHE_ROOT}/${name}" \
        --output-dir "${RESULT_ROOT}/figures" --seed 42
      python scripts/plot_tsne_task_a.py \
        --feature-dir "${CACHE_ROOT}/${name}" \
        --output-dir "${RESULT_ROOT}/figures" --task B --seed 42
      python scripts/audit_task_b_temporal_confounding.py \
        --feature-dir "${CACHE_ROOT}/${name}" \
        --output-dir "${RESULT_ROOT}/temporal_audit" \
        --purge-gap-groups 1 --seed 42
    fi
    rm -rf "${CACHE_ROOT:?}/${name}"
  done
done

for profile in psd_classification annotation_classification; do
  name="psd_${profile}_w5_g10"
  python scripts/extract_psd_features.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/${name}" \
    --task B --preprocessing "${profile}" --montage edf32 \
    --window-sec 5 --stride-sec 5 --filter-scope group_bounded \
    --transition-guard-sec 10 --subjects "${SUBJECTS[@]}"
  python scripts/run_classical.py \
    --feature-dir "${CACHE_ROOT}/${name}" \
    --output-dir "${EXPERIMENT_ROOT}/${name}" \
    --project-root . --task B --model psd_svm \
    --protocol within_subject --subject-allowlist "${ALLOWLIST}"
  rm -rf "${CACHE_ROOT:?}/${name}"
done

for profile in physiological_validation physiological_global_zscore annotation_global_zscore annotation_microvolt; do
  name="eegnet_${profile}_w5_g0"
  python scripts/extract_eeg_windows.py \
    --raw-root "${RAW_ROOT}" --output-dir "${CACHE_ROOT}/${name}" \
    --task B --preprocessing "${profile}" --montage edf32 \
    --window-sec 5 --stride-sec 5 --filter-scope group_bounded \
    --transition-guard-sec 0 --subjects "${SUBJECTS[@]}"
  python scripts/run_deep.py \
    --cache-dir "${CACHE_ROOT}/${name}" \
    --output-dir "${EXPERIMENT_ROOT}/${name}" \
    --project-root . --task B --model eegnet \
    --protocol within_subject --subject-allowlist "${ALLOWLIST}" \
    --device cuda --epochs 20 --patience 5 --batch-size 128 \
    --num-workers 4 --seed 42
  rm -rf "${CACHE_ROOT:?}/${name}"
done

python scripts/assemble_task_b_hypotheses.py \
  --experiments-root "${EXPERIMENT_ROOT}" \
  --output-dir "${RESULT_ROOT}"
rm -rf "${CACHE_ROOT}"
echo "TASK_B_HYPOTHESIS_SMOKE_COMPLETE"
