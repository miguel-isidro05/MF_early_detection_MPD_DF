# Task A Within-Subject Audit

Date: 2026-07-28

Audited snapshot: `task_a_within_review_snapshot`

Code revision reported by the snapshot: `8179bdf`

## Decision

The snapshot is useful preliminary evidence, but it is not a final or
publishable result bundle. The grouped within-subject design has no detected
direct split leakage, and EEGNet shows a meaningful signal. However, the
classical PSD representation did not match the declared logarithmic feature
protocol, the SVM emitted repeated convergence warnings, and the EEGNet outer
refit used the early-stopping history length instead of the best validation
epoch. All three within-subject models must therefore be rerun.

The applied, unseen-participant claim also remains untested because the LOSO
run stopped at its first outer fold.

## Integrity checks

- 291,413 out-of-fold windows shared by all three models.
- 42 eligible participants and 205 outer folds.
- No duplicate `(subject, window_start_sec)` prediction keys.
- No train/test annotation-group overlap.
- Identical test assignments and prediction roster across models.
- Official alignment report status: `MATCH`.
- Eight within-subject exclusions are documented in
  `data/metadata/excluded_within_task_a.csv`.

## Preliminary subject-macro results

| Model | Balanced accuracy | Fatigue1 F1 | Recall | Precision |
|---|---:|---:|---:|---:|
| PSD-SVM | 0.695 | 0.494 | 0.645 | 0.416 |
| Random Forest | 0.656 | 0.460 | 0.605 | 0.391 |
| EEGNet | 0.718 | 0.520 | 0.696 | 0.443 |

These values are descriptive only because the corrected rerun can change
them. Window-pooled metrics must not be presented as primary results: the
macro participant F1 is approximately 0.09 to 0.11 lower than pooled F1.
Accuracy is also misleading because the negative-class prevalence gives a
78.7% majority-class accuracy baseline.

## Confirmed defects and corrections

1. The preliminary SVM emitted 628 convergence warnings before within-subject
   completion. The corrected runner uses `max_iter=20000` and converts any
   convergence warning into a run failure.
2. The declared PSD protocol required logarithmic power, but the preliminary
   cache contained raw absolute powers and raw ratios. Corrected features use
   `10*log10(power)` and `10*log10(ratio)`.
3. The EEGNet parameter name `f1` collided with the F1 metric column.
   Corrected outputs use `eegnet_f1`.
4. The preliminary EEGNet refit trained for the complete early-stopping
   history length. Corrected refits use the epoch attaining minimum finite
   validation loss and record `selected_epochs`.
5. Linear SVM decision values are not probabilities. The publication
   assembler therefore excludes SVM from probability-reliability plots.

The preliminary outputs are archived rather than overwritten. These changes
are protocol corrections identified by audit, not post-hoc score selection.

## Publication gate

Required before a model-performance paper:

- corrected within-subject rerun;
- complete nested LOSO evaluation;
- `edf32` versus `paper28` channel ablation;
- participant-bootstrap confidence intervals and paired multiplicity-adjusted
  tests;
- final tables, OOF figures, and complete provenance bundle;
- explicit claim that Task A is Wakefulness versus Fatigue1 concurrent state
  recognition.

Required before claiming operational early detection:

- chronological or purged temporal sensitivity analysis;
- episode sensitivity, detection latency or lead time, and false alarms/hour;
- robustness across multiple EEGNet seeds;
- a deployment-relevant threshold or calibration procedure learned without
  test-participant labels.

The dataset paper's Table 9 EEG result is not a direct benchmark because its
split and aggregation details are not equivalent to the nested grouped
protocol. It may be cited as context, not as a like-for-like reproduction
target.
