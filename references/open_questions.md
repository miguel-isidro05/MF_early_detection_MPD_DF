# Open Questions

1. What exact train/validation/test split produced Table 9?
2. Were Table 9 metrics macro-averaged, weighted, pooled, or fatigue-class binary metrics?
3. Did Table 9 use all 32 EDF channels or the 28 analytical channels listed in the descriptor?
4. Which preprocessing pipeline, normalization scope, artifact policy, and overlap were used for MSCNN-CAM?
5. Is the cited 2024 MSCNN-CAM source code available from the authors?
6. Were 1-second samples balanced, weighted, or undersampled during training?
7. Did the original experiment preserve participant and 30-second block boundaries?

Until resolved, choices affecting these questions must be labeled:

> Inferred implementation choice — not explicitly reported in the reference paper.

