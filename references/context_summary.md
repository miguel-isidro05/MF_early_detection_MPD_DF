# Context Summary

MPD-DF contains approximately two-hour simulated-driving recordings from 50 participants, including 32-channel EEG at 500 Hz and physician change-point annotations. The current project uses EEG as the only model input.

The descriptor defines five physiological states: Wakefulness (0), Fatigue1 (1), Fatigue2 (2), Fatigue3 (3), and Fatigue4 (4). Labels 8 and 9 denote signal abnormality and severe artifacts. Fatigue1 is the earliest annotated fatigue stage and is therefore the primary positive class for the proposed paper.

The descriptor reports an aggregate binary MSCNN-CAM result for 1-second EEG: Accuracy 0.876, Precision 0.760, Recall 0.699, and F1 0.705. The public repository contains alignment code only; it does not contain the classifier, training loop, split assignments, or evaluation script.

The supplied related papers support EEG spectral and cross-subject fatigue analysis, but their datasets and fatigue definitions are not interchangeable with MPD-DF. They are methodological context, not direct performance anchors.

The ST-SODE paper is the closest cross-subject methodological reference. It evaluates SEED-VIG (23 participants, continuous PERCLOS labels) and a private eight-person N-Back dataset. It explicitly contrasts LOSO with within-subject five-fold validation and shows that personalized performance is substantially more optimistic. Its 0.5-second epochs and 5-20-second aggregation windows are not reference settings for MPD-DF.

Wang et al. (2022) use 14-channel Emotiv EEG from 10 participants, subjective SOFI-C scores, selected 5-second clean segments, basic-scale entropy, and MVAR-PSI connectivity. The study supports spectral/connectivity interpretation but provides no subject-independent benchmark comparable to MPD-DF.

The supplied 2026 wearable-biosensor review is secondary context. Its proposed IoT framework is conceptual and unvalidated, and some broad epidemiological and commercial claims require primary-source verification. It is not a performance anchor for this project.
