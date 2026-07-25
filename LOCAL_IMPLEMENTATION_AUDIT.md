# Local Implementation Audit

| Component | Source | Status | Validation |
|---|---|---|---|
| Official alignment | `Dataset/MPD-DF-main/DataAlign.py` | Reimplemented from headers and timestamps | Exact 50-subject count match |
| Portable EEG alignment | `src/mpd_df/dataset.py` | PSG timing frozen in CSV; no PSG waveform needed | Portable subset equals full raw tree |
| EDF unit handling | `src/mpd_df/dataset.py` | Stored values treated as µV, converted to V | Full signal audit and real EDF smoke |
| Annotation parser | `src/mpd_df/annotations.py` | Labels 0-4, 8, 9; timestamp expansion | Unit tests and exact aggregate counts |
| Evaluation groups | `src/mpd_df/splits.py` | Fixed 30 s temporal bins | Group-disjoint tests |
| Strict filtering | `iter_preprocessed_windows()` | Group-bounded for within-subject; subject-continuous for LOSO | Real EDF feature smoke |
| PSD features | `src/mpd_df/features.py` | Band powers, relative powers, ratios, entropy | Unit and end-to-end smoke tests |
| Classical models | `src/mpd_df/models/classical.py` | Linear SVM and Random Forest | Grouped real-data smoke |
| EEGNet | `src/mpd_df/models/deep.py` | Standardized local baseline | Real EDF training/checkpoint smoke |
| MSCNN-CAM | `src/mpd_df/models/deep.py` | Inferred local implementation | Real EDF training/checkpoint smoke; not official-equivalent |
| Deep runner | `scripts/run_deep.py` | Early stopping, class weighting, checkpoints, subject metrics | EEGNet and MSCNN-CAM one-epoch smoke |
| RTX environment | `environment/RTX4060_SETUP.md` | CUDA wheel and runtime preflight | Must be verified on the RTX host |

FatigueSet-specific labels, channel assumptions, and result claims are not reused.
