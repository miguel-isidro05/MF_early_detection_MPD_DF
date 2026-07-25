# Methodological Constraints

- Physician decisions are represented as change points over original 30-second blocks. One-second windows from the same block are dependent.
- Random window-level cross-validation is prohibited for primary results.
- Within-subject evaluation must group by original annotation block or a larger contiguous unit.
- Final evaluation is Leave-One-Subject-Out; no held-out subject may influence scaling, channel selection, ICA, tuning, balancing, or threshold selection.
- Labels 8 and 9 are excluded from clean-state tasks unless a reference-matching artifact policy is explicitly evaluated.
- The EDF montage has 32 channels, while the descriptor lists 28 analytical channels in one section. This discrepancy remains an explicit reproduction variable.
- EDF physical-dimension bytes encode a corrupted/unrecognized microvolt string (`¦ÌV`), which MNE exposes as `n/a` and otherwise treats numerically as volts. The loader applies an explicit `1e-6` scale to interpret stored values as microvolts. This inference must remain documented and sensitivity-checked.
- The descriptor gives two figure-oriented preprocessing pipelines, neither proven to be the Table 9 classification pipeline.
- Fatigue3 and Fatigue4 are not suitable as primary subject-independent classes because very few participants reach those states.
