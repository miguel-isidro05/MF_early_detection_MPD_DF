# Decisions

## D001: Project Location

The project lives under `code/mpd_df_project/`.

## D002: EEG-Only Working Data

`data/eeg_only/` links EEG, annotations, and questionnaire metadata without duplicating the full dataset.

## D003: Seed

The global seed is `42`.

## D004: Label Timing

Annotation timestamps define label intervals, as in `DataAlign.py`. The auxiliary block-number column is retained for audit only because it is approximate and can repeat.

## D005: Grouping

Within-subject folds use fixed contiguous 30-second groups anchored to the aligned EEG interval. This is conservative and reproducible when the source block column is ambiguous.

## D006: EEG Units

Stored EEG numbers are interpreted as microvolts and scaled by `1e-6` to volts because the EDF unit bytes are malformed.

## D007: Reproduction Claims

The local MSCNN-CAM is a reimplementation. It cannot support an exact-reproduction claim without the cited architecture and Table 9 protocol.

