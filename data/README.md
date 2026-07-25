# Data layout

`raw/mpd_df_raw_full` and `eeg_only/` are local symlinks. Git does not track them.

The portable RTX archive contains:

```text
MPD_DF_EEG_ONLY/
├── EEG/                     # 50 EDF files
├── Annotation/              # 50 physician annotation files
├── alignment_manifest.csv   # PSG-derived overlap timing
├── bundle_manifest.json
└── README.md
```

The archive excludes PSG, ECG, EOG, respiratory signals, questionnaire data, and
derived caches. After extraction, pass `MPD_DF_EEG_ONLY` as `--raw-root`.

`metadata/` stores audit outputs. `derived/` is reserved for feature and memmap
caches and is ignored by Git.
