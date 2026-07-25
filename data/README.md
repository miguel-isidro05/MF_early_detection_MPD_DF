# Data Layout

The full MPD-DF raw dataset is not duplicated here.

`raw/mpd_df_raw_full` is a symlink to the downloaded full dataset.

`eeg_only/` contains symlinks to only the inputs needed for EEG modeling and audit:

- `EEG/`: 50 raw EEG EDF files.
- `Annotation/`: 50 physician annotation files.
- `Questionnaire information.xlsx`: questionnaire metadata.

Derived arrays, inventories, caches, and experiment-ready epochs should be written under `metadata/`, `derived/`, or experiment folders, never into the source dataset.
