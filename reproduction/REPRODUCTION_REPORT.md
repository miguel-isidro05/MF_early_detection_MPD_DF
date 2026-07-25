# MPD-DF EEG Reproduction Report

## Status

`DATA_AND_FIGURE_REPRODUCTION_COMPLETE; TABLE9_EXACT_REPRODUCTION_BLOCKED`

The public alignment, dataset audit, label analysis, and EEG figure workflows have
been reproduced. The classification result in Table 9 cannot be reproduced exactly
from the public material because its executable protocol is not published.

## Table 9 reference

| Input | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1-second EEG | 0.876 | 0.760 | 0.699 | 0.705 |

No local metric is presented as a reproduced Table 9 value.

## Verified data path

- Dataset: 50 raw MPD-DF EEG recordings.
- Sampling: 500 Hz, 32 EDF channels.
- Alignment: latest EEG/PSG/annotation start, earliest EEG/PSG end, final one-second
  trim.
- Result: 372,404 seconds and zero aggregate label-count differences from an
  independent literal implementation of `DataAlign.py`.
- Artifacts: labels 8 and 9 remain explicit and are excluded from clean-state
  classification tasks.

## Table 7

The local 118-minute normalization does not equal the published counts.

| Label | Local 1 s | Paper 1 s | Difference |
|---|---:|---:|---:|
| Wakefulness | 268,404 | 266,718 | +1,686 |
| Fatigue1 | 60,890 | 60,292 | +598 |
| Fatigue2 | 14,351 | 14,789 | -438 |
| Fatigue3 | 760 | 760 | 0 |
| Fatigue4 | 0 | 0 | 0 |

The published 10-second Wakefulness row contains 26,716 non-overlapping windows,
which would require 267,160 one-second samples, but the same table reports only
266,718. That row is internally inconsistent if all strategies use the same
normalized pool. Full comparisons are in
`reproduction/tables/table07_full_comparison.csv`.

## Figures

- Figure 6: all five fatigue states plus Signal Abnormality and Severe Artifacts;
  30-second EEG screens.
- Figure 7: beta, alpha, theta, and delta waveforms. Published x-axis starts were
  recovered from the figure; subject and channel choices remain inferred.
- Figure 8: label timeline, participant totals, and the 7200-second detail.
- Figure 10: Fp1, C3, T7, and O1 for all 50 participants using the published
  visualization preprocessing.
- Figure 11: Participant 10 topographies with the published layout and a shared
  PSD scale. This remains a no-ICA diagnostic because the paper does not publish
  removed components, rejection criteria, or the exact PSD estimator.

Every generated figure has CSV, JSON, or NPZ source data under
`reproduction/tables/`.

## Classification boundary

The paper and official repository do not provide:

- Table 9 splits or folds;
- classification preprocessing and normalization scope;
- 32-channel versus 28-channel input;
- metric averaging and positive-class convention;
- artifact and balancing policy;
- MSCNN-CAM training code and complete hyperparameters.

`configs/reproduction/r1_inferred_matrix.yaml` defines executable sensitivity
variants for the RTX 4060. Their results must be labeled inferred reimplementations.
