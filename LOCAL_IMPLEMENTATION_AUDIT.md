# Local Implementation Audit

This audit distinguishes reusable engineering from FatigueSet-specific scientific assumptions.

| Component | Existing path | Current purpose | Reusable | Required changes | Validation performed |
|---|---|---|---|---|---|
| Non-finite repair | `DataAnalysis/fatigue_eeg/io.py` | Repair sparse NaN/Inf values | Yes | Calibrate thresholds on MPD-DF | Existing unit tests reviewed; MPD-DF signal QC pending |
| Band-power features | `DataAnalysis/fatigue_eeg/features.py` | PSD features and ratios | Yes | Pass MPD-DF channels, bands, and sampling rate | Contract reproduced in new unit-tested module |
| Classical pipelines | `DataAnalysis/fatigue_eeg/modeling.py` | Scaled sklearn baselines | Yes | Use MPD-DF tasks and grouped splits | Leakage-safe scaling pattern retained |
| LOSO evaluation | `DataAnalysis/fatigue_eeg/evaluation.py` | Subject-wise evaluation and aggregation | Yes | Remove 12-subject/FatigueSet guards | Split invariants implemented and tested |
| Statistical tests | `DataAnalysis/fatigue_eeg/statistics.py` | Wilcoxon and Holm correction | Yes | Define planned MPD-DF comparisons | Code reviewed; integration pending results |
| EEGNet | `DataAnalysis/fatigue_eeg/deep_modeling.py` | Binary EEG classifier wrapper | Partial | Parameterize 32/28 channels and 1 s inputs | New tensor-shape smoke test passes |
| Advanced models | `DataAnalysis/fatigue_eeg/advanced_modeling.py` | CSP, Riemannian and adaptation methods | Optional | Keep outside mandatory baseline set | Read-only audit completed |
| FatigueSet loader | `DataAnalysis/fatigue_eeg/io.py` | Muse CSV ingestion | No | Replace with EDF and annotation loader | Replaced by MPD-DF-specific loader |
| FatigueSet epoching | `DataAnalysis/fatigue_eeg/epochs.py` | Stage-marker-based 3 s epochs | No | Replace with physician labels and 30 s groups | Replaced by block-preserving index |
| BIDS/MOABB adapters | `DataAnalysis/fatigue_eeg/bids.py`, `moabb_adapter.py` | Muse dataset standardization | Partial | Implement subject-indexed MPD-DF contract | Local subject/index API implemented |
| Deep experiment runner | `DataAnalysis/scripts/run_deep_cross_subject.py` | Checkpointed LOSO training | Partial | Remove fixed subject/channel/task assumptions | Integration pending |
| Official aligner | `official_code/MPD-DF-main/DataAlign.py` | EEG/PSG/annotation alignment | Yes, as reference | Parameterize paths; avoid loading unused PSG signals | Header-equivalent alignment implemented; numerical comparison pending |

New code must not import conclusions, thresholds, channel assumptions, or labels from FatigueSet.

