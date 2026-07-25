# Smoke Test Report

Date: 2026-07-25  
Seed: 42  
Host: macOS arm64, Python 3.11.15

## Completed checks

| Check | Result |
|---|---|
| Unit and contract tests | 23 passed |
| Full 50-subject EEG scan | 50 passed, no NaN/Inf |
| Official label alignment | MATCH, 372,404 seconds |
| Portable alignment without PSG | MATCH against the full raw tree |
| PSD extraction from real EDF | Passed |
| Grouped PSD+SVM evaluation | Passed |
| EEG memmap cache | Passed, 28 channels at 200 Hz |
| EEGNet train/checkpoint/predict | Passed, one smoke fold |
| Local MSCNN-CAM train/checkpoint/predict | Passed, one smoke fold |
| Figures 6, 7, 10, 11 | Regenerated and visually inspected |
| EEG-only ZIP | 103 entries, CRC passed, SHA-256 recorded |
| Full nested split preflight | A: 42 eligible; B/C: 45 eligible |

The smoke metrics are not scientific results. Each deep smoke used one participant,
128 balanced windows, one epoch, and one fold. Their purpose was to exercise I/O,
preprocessing, splitting, optimization, checkpoint loading, and prediction.

## Tooling note

`pytest-cov` and `ruff` are declared development dependencies but are not installed
in the current conda environment. Plain `pytest` passed. Syntax was also exercised
through test imports and script execution. Direct `compileall` inside the project
could not write bytecode because this session mounts the tree read-only for shell
processes; compilation against a temporary copy is used in final verification.
