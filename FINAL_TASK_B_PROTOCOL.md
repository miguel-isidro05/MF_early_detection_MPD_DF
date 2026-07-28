# Final Task B Protocol

Task B is the user-selected primary application task:

```text
Class 0: Wakefulness
Class 1: Fatigue1 + Fatigue2
```

It operationalizes Fatigue1 as onset fatigue and Fatigue2 as the following
drowsiness-related stage, using the definitions in the MPD-DF descriptor
([Li et al., Scientific Data, 2026](https://doi.org/10.1038/s41597-026-06634-4)).
It is broader than the original Task A (`0` versus `1`) and narrower than the
reference paper's aggregate binary task (`0` versus `1+2+3+4`). Task A remains
a strict sensitivity analysis. Task C and MSCNN-CAM are excluded by study
decision.

The change is not justified by higher accuracy. The operational rationale is
that labels 1 and 2 precede the paper's sleep-related Fatigue3/Fatigue4 stages.
This grouping is a study decision rather than an exact reproduction of the
paper's aggregate binary experiment.

## Primary benchmark

- Non-overlapping 1-second windows, matching the reported EEG input duration.
- Windows never cross an annotation change or fixed 30-second group.
- Signal Abnormality and Severe Artifacts are excluded.
- EDF32 is primary; paper28 is the channel sensitivity analysis.
- Bandpass 1-100 Hz, 50 Hz notch, channel-mean removal, and resampling to
  200 Hz.
- PSD features use absolute power in dB, relative power, fatigue-related
  ratios, and spectral entropy. Scaling is fitted inside each training fold.
- EEGNet uses per-window/per-channel z-score only in the paper-compatible
  primary benchmark.
- Within-subject grouped validation is developmental. LOSO is the primary
  unseen-participant evaluation.
- Hyperparameters and thresholds are selected only from inner validation.

## Preprocessing sensitivity matrix

Before the full benchmark, a five-subject execution and sensitivity screen
compares:

1. 1-100 Hz paper-compatible filtering with and without per-window z-score.
2. 0.3-35 Hz annotation-compatible filtering.
3. Per-channel z-score, global-window z-score, and microvolt scaling.
4. Window durations of 1, 5, and 10 seconds.
5. Inclusion versus a 10-second guard around label transitions.

The primary profile above is fixed before this screen. Outer-fold screen
metrics cannot select or modify the primary pipeline. They are reported only
as a small-cohort sensitivity analysis and execution check.

## Physiological validation

The final bundle reports Wakefulness, Fatigue1, and Fatigue2 separately for
physiology even though labels 1 and 2 are pooled for classification. Required
outputs are:

- paired subject-level absolute and relative band-power changes;
- channel-by-band effect-size heatmap;
- theta/alpha/beta/delta effect-size topographies;
- regional progression from Wakefulness to Fatigue2;
- source tables, bootstrap confidence intervals, and corrected tests.

Raw PSD magnitude topographies from Participant 10 remain reproduction
figures. Effect-size topographies are the inferential figures for the new
paper.
