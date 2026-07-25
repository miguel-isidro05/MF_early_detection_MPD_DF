# Relevant Baselines

Mandatory baselines for the project:

| Model | Representation | Role |
|---|---|---|
| PSD + SVM | Absolute/relative band power, ratios, spectral entropy | Interpretable classical baseline |
| Random Forest | Same fixed PSD feature set | Nonlinear classical baseline |
| EEGNet | Minimally processed EEG | Compact learned baseline |
| MSCNN-CAM | Minimally processed EEG | Reference-family model |

The local MSCNN-CAM module is a documented reimplementation, not an exact copy of the cited 2024 model. Exact-equivalence claims remain blocked until the cited architecture and training details are obtained and verified.

Contextual methods not in the mandatory benchmark:

| Method | Dataset/protocol | Relevance boundary |
|---|---|---|
| ST-SODE | SEED-VIG and private N-Back; LOSO | Motivates cross-subject spectral dynamics; labels and windows differ |
| DE-SVM / PSD-SVM | Within-subject and LOSO comparisons in the ST-SODE study | Supports reporting the personalized-generalization gap |
| Basic-scale entropy + MVAR-PSI | 10-person simulated driving, SOFI-C | Descriptive mechanism only; no comparable LOSO classification |

