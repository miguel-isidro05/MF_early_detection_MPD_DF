# RTX 4060 Execution Plan

## 1. Preflight

1. Extract `packages/mpd_df_eeg_only_rtx.zip`.
2. Create the environment from `environment/RTX4060_SETUP.md`.
3. Confirm `torch.cuda.is_available()` and the RTX 4060 device name.
4. Run all tests.
5. Verify the extracted dataset against `alignment_manifest.csv`.

## 2. Reproduction variants

Run the matrix in `configs/reproduction/r1_inferred_matrix.yaml`. These runs test
the unresolved 28/32-channel and preprocessing choices. They are inferred R1
variants, not an exact official pipeline.

Within-subject comparisons must use the task-specific allowlist under
`data/metadata/eligible_within_task_*.txt` for every model. Nested grouped
validation leaves 42 eligible participants for Task A and 45 for Tasks B/C.
Exclusion reasons are stored in the adjacent CSV files. LOSO keeps all subjects.

## 3. Standardized study

Run within-subject grouped validation first. Then run LOSO only after the pipeline,
logs, and initial ablations are stable. The planned matrix is in
`configs/rtx4060_experiment_matrix.yaml`.

Every run must save:

- resolved configuration and seed;
- code commit and environment;
- train, validation, and test subjects;
- split assignments;
- predictions and subject-level metrics;
- checkpoints and learning curves;
- CUDA, cuDNN, driver, and GPU identity.

Do not promote smoke metrics or inferred R1 metrics to Table 9 reproduction results.
