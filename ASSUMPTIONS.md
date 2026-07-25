# Assumptions

- Global seed: `42`.
- The supplied Figshare folder is the raw MPD-DF release.
- EEG is the only model input.
- PSG timing metadata may be read to preserve official alignment.
- Annotation timestamps are authoritative; the second column is auxiliary.
- The EDF's malformed physical-dimension text was intended to denote microvolts.

Any consequential method choice absent from the paper must be recorded as:

> Inferred implementation choice — not explicitly reported in the reference paper.

