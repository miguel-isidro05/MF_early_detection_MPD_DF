# Issues

## Open Critical Reproduction Gaps

- Table 9 split, averaging, channels, preprocessing, balancing, and model implementation are unreported.
- The cited 2024 MSCNN-CAM full text and code are not present locally.

## Verified Dataset Discrepancies

- The EDF unit bytes are malformed (`¦ÌV`); the loader applies an explicit microvolt interpretation.
- The annotation block-number column is approximate and repeats 28 times. Timestamps are the authoritative label boundaries, matching `DataAlign.py`.
- Literal 118-minute normalization does not exactly reproduce Table 7:
  - Wakefulness: +1,686 s.
  - Fatigue1: +598 s.
  - Fatigue2: -438 s.
  - Fatigue3 and Fatigue4: exact.
- The EDF has 32 channels; a descriptor section lists 28 analytical channels.

## Deferred

- IEEE template will be added by the user at the end.
- Figure 11 ICA component rejection is not reproducible from the paper because removed components and criteria are absent.

