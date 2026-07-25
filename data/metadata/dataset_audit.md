# MPD-DF Dataset Audit

```json
{
  "subjects": 50,
  "successful": 50,
  "errors": [],
  "unique_eeg_channel_count": [
    32
  ],
  "unique_sampling_rates": [
    500.0
  ],
  "edf_channel_order_matches_verified_32": true,
  "paper_comparison_note": "Both complete official-overlap counts and a literal 118-minute normalization are reported. The remaining differences from Table 7 are unresolved and are not corrected.",
  "class_comparison": [
    {
      "label": 0,
      "local_aligned_seconds": 278545,
      "local_normalized_118min_seconds": 268404,
      "paper_normalized_seconds": 266718,
      "normalized_difference": 1686
    },
    {
      "label": 1,
      "local_aligned_seconds": 64001,
      "local_normalized_118min_seconds": 60890,
      "paper_normalized_seconds": 60292,
      "normalized_difference": 598
    },
    {
      "label": 2,
      "local_aligned_seconds": 19973,
      "local_normalized_118min_seconds": 14351,
      "paper_normalized_seconds": 14789,
      "normalized_difference": -438
    },
    {
      "label": 3,
      "local_aligned_seconds": 1498,
      "local_normalized_118min_seconds": 760,
      "paper_normalized_seconds": 760,
      "normalized_difference": 0
    },
    {
      "label": 4,
      "local_aligned_seconds": 310,
      "local_normalized_118min_seconds": 0,
      "paper_normalized_seconds": 0,
      "normalized_difference": 0
    }
  ]
}
```
