# R0 Official Pipeline Audit

The public repository contains:

- `README.md`
- `DataAlign.py`
- `requirements.txt`
- `LICENSE`

`DataAlign.py` reads EEG and PSG EDF files, aligns them with annotation timestamps, creates one-second epochs, and writes MATLAB files. It does not contain:

- MSCNN-CAM;
- any classifier;
- training or evaluation loops;
- split assignments;
- Table 9 metric code;
- model hyperparameters.

The repository hard-codes `/data/MPDDF/rawdata` and an output path. The local implementation preserves the alignment semantics through configurable paths and reads PSG timing metadata only; PSG values are never used as model inputs.

The official `requirements.txt` incorrectly attempts to install Python standard-library modules `logging` and `re`. These entries are not copied into the project environment.

