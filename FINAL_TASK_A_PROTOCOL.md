# Final Task A Protocol

This is the final study protocol for Task A only: Wakefulness versus
Fatigue1. MSCNN-CAM, Tasks B/C, and ICA-cleaned classification are outside
this execution by explicit study decision.

## Fixed input and segmentation

- EEG only, 32 verified EDF channels (`edf32`) at 500 Hz.
- Exclude Signal Abnormality and Severe Artifacts; Task A retains only labels
  Wakefulness and Fatigue1.
- Align labels with the frozen official timing manifest.
- Use non-overlapping 1-second windows. A window never crosses its 30-second
  physician-annotation bin.
- No random window split is permitted.

## Standard classification preprocessing

The Figure 10 pipeline is retained for descriptive visualization, but it is
not silently used as the PSD classification representation.

1. Convert EDF stored values from microvolts to volts.
2. Zero-phase 1-100 Hz bandpass filter.
3. Zero-phase 50 Hz notch filter.
4. Remove the channel mean.
5. Resample to 200 Hz.
6. Segment into the fixed 1-second windows.

For EEGNet, normalize each window and channel with its own z-score. This has
no fitted statistic from the held-out participant. For PSD features, retain
the filtered physical-amplitude signal, express absolute band powers and power
ratios as `10*log10` values, and fit the feature scaler on the training fold
only. ICA is not part of the
primary classifier; the manual EEGLAB/PSD reproduction remains separate.

Within-subject filtering is group-bounded so a zero-phase filter cannot mix
train and test annotation bins. LOSO filtering is subject-continuous because
the held-out participant is never included in training.

## Evaluation and selection

### Development / personalized evaluation

For each feasible participant, run five-fold StratifiedGroupKFold using fixed
30-second annotation bins as groups. Report out-of-fold subject metrics and
then mean plus standard deviation across participants.

### Main subject-independent evaluation

Use LOSO. Each outer test participant remains untouched. On the remaining
participants, tune the model and decision threshold by a five-fold
participant-grouped inner validation. Refit the selected candidate on the
entire outer-training set and evaluate once on the held-out participant.
For EEGNet, the refit epoch count is the one-based epoch attaining the minimum
validation loss, not the length of the early-stopping history.

The selection order is fatigue-class F1, then fatigue recall, Cohen's kappa,
and balanced accuracy. Accuracy is descriptive only.

## Models and candidate sets

The candidate sets are deliberately small and justified; they are not a broad
score-chasing sweep.

| Model | Inner candidates |
|---|---|
| PSD-SVM | Linear SVM, `C` in {0.1, 1, 10}, balanced class weight, `max_iter=20000`; non-convergence aborts the run |
| Random Forest | 500 trees; max depth in {12, none}; min samples leaf in {1, 5}; balanced subsample class weight |
| EEGNet | `F1` in {8, 16}; dropout in {0.25, 0.50}; learning rate in {3e-4, 1e-3}; AdamW, weight decay 1e-4, early stopping on inner validation |

The EEGNet candidate set is represented by four prespecified combinations,
not the Cartesian product of every value. Inner selection is computationally
expensive but preserves the held-out LOSO participant as a true test subject.

## Required Task A comparisons

1. Main model comparison: PSD-SVM, Random Forest, EEGNet under LOSO.
2. Protocol comparison: within-subject grouped versus LOSO.
3. Representation comparison: tuned PSD-SVM versus tuned EEGNet.
4. Channel ablation: full `edf32` versus `paper28`, using the selected
   primary model under LOSO.

## Required deliverables

Tables: protocol/preprocessing, selected hyperparameters, within-subject
metrics, LOSO metrics, per-subject metrics, channel ablation, and Wilcoxon
comparisons.

Figures: framework diagram; label distribution; existing signal/waveform and
representative-channel reproductions; manual PSD topography; within-subject
versus LOSO comparison; normalized and count confusion matrices; per-subject
F1/recall; RF/EEGNet probability-reliability curves; EEGNet learning curves; representation and
channel-ablation plots; and exploratory t-SNE of the training-fold-scaled PSD
features. t-SNE is explicitly exploratory.
