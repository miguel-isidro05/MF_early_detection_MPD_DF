# Preprocessing Protocol

The MPD-DF paper reports two distinct EEG visualization procedures. It does not
publish the exact preprocessing, splits, or full hyperparameters used for Table 9.
The benchmark runner therefore records a standardized local protocol rather than
claiming an exact Table 9 reproduction.

## Figure 6 and Figure 7: annotation visualization

Published procedure: EEGLAB, 0.3-35 Hz bandpass and 49-51 Hz notch/band-stop.
The implemented `annotation_visualization` profile applies a zero-phase fourth
order 0.3-35 Hz bandpass and a zero-phase fourth order 49-51 Hz band-stop at the
native 500 Hz sampling rate. No z-score normalization or resampling is applied.

Expected appearance: 30-second multichannel traces retain physical amplitude and
show alpha slowing, sleep spindles, and K-complex-like events when present. The
precise participant/window selection in the paper is not published, so line-by-line
pixel agreement is not expected.

## Figure 10: physiological data validation

Published EEG procedure: 1-100 Hz bandpass, 50 Hz notch, mean removal,
downsampling to 200 Hz, and z-score normalization. The implemented
`physiological_validation` profile follows those operations. In Figure 10,
z-score is applied per displayed 10-second channel segment because the paper does
not disclose its normalization scope.

Expected appearance: each trace is dimensionless and centered near zero before
plot stacking. Differences in color, vertical offsets, and exact waveform windows
are display choices, not evidence of different signal preprocessing.

## Figure 11: PSD topographies

Published procedure: Figure 6 preprocessing plus EEGLAB ICA to remove ocular and
muscular components before average PSD topographies. The public material does not
identify ICA components, rejection thresholds, PSD estimator, or segment choices.
`plot_psd_topographies.py` intentionally produces a clearly labeled **no-ICA
diagnostic**, not an exact Figure 11 reproduction. Its values and color pattern
must not be used to validate or claim exact agreement with the published map.

## Classification benchmark: Task A and B

The paper states only that 1-second EEG was preprocessed and fed into MSCNN-CAM
using parameters from a separate cited study. It omits the classification filter,
normalization scope, montage, split, and balancing policy. The RTX benchmark uses
`physiological_validation` as a declared, reproducible standard: 1-100 Hz, 50 Hz
notch, per-window demean, 200 Hz resampling, and per-window/channel z-score.

The resulting Task A/B benchmark is valid for comparing local models under the
same protocol; it is not an exact Table 9 result.
