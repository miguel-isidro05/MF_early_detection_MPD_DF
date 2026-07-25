# MPD-DF EEG Reproduction Report

## Status

`NOT_REPRODUCED`

R0 alignment inspection is implemented. R1 MSCNN-CAM training has not yet produced a valid reference-comparable result.

## Reference

| Input | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1-second EEG | 0.876 | 0.760 | 0.699 | 0.705 |

## Verified Task Elements

- Dataset: local raw MPD-DF, 50 participants.
- Signal: EEG at 500 Hz.
- Task C: Wakefulness versus Fatigue1+Fatigue2+Fatigue3+Fatigue4.
- Segmentation: 1-second windows, initially without overlap.
- Artifact labels: 8 and 9 remain separate and are excluded from clean-state tasks.
- Alignment: latest EEG/PSG/annotation start and earliest EEG/PSG end; final one-second trim follows `DataAlign.py`.

## Preprocessing Comparison

| Pipeline | Purpose in paper | Bandpass | Notch | Downsample | Normalization | ICA |
|---|---|---:|---:|---:|---|---|
| Annotation-oriented visualization | Figures 6-8 | 0.3-35 Hz | 49-51 Hz | Not reported | Not reported | Not necessarily |
| Physiological validation | Figure 10 | 1-100 Hz | 50 Hz | 200 Hz | z-score, scope unreported | No |
| Participant 10 topographies | Figure 11 | Based on prior figure pipeline | Verify | Verify | Verify | Yes, EEGLAB |
| MSCNN-CAM classification | Table 9 | Unreported | Unreported | Unreported | Unreported | Unreported |

## Critical Comparability Gaps

The descriptor and public repository do not provide:

- Table 9 split assignments or fold design;
- metric averaging and positive-class convention;
- exact 32-versus-28-channel input;
- classification preprocessing and normalization scope;
- artifact policy and class balancing;
- MSCNN-CAM source code or complete hyperparameters.

The local MSCNN-CAM module is therefore a documented reimplementation. Any result from it is R1/R2 evidence, not an exact official reproduction.

## Results

No valid metrics are recorded yet.

## Recreated Figures

- Figure 6 functional surrogate: five 30-second Participant 10 EEG states. Subject and segment selection are explicitly inferred.
- Figure 7 functional surrogate: delta/theta/alpha/beta waveforms from O1. Channel and segment selection are explicitly inferred.
- Figure 8 functional recreation: complete label timeline, normalized participant distribution, and 7200-second zoom.
- Figure 10 EEG-only panels: Fp1, C3, T7, and O1 for all participants. The 10-second selection and z-score scope are explicitly inferred.
- Figure 11 diagnostic only: Participant 10 no-ICA integrated PSD topographies. This is not accepted as the reference reproduction because the paper omits ICA component decisions and exact PSD settings.

Every figure has machine-readable source data under `reproduction/tables/`.

## Software And Hardware

Environment capture is generated per experiment. The current validated development environment is Python 3.11 with MNE 1.12.1, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3, scikit-learn 1.9.0, and PyTorch 2.12.1.
