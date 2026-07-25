# Code Review Report

## Corrected

1. The original feature cache filtered continuous chunks before grouped
   within-subject splitting. Zero-phase filtering could mix signal support across
   adjacent evaluation groups. The strict path now uses group-bounded filtering.
2. Feature fingerprints included absolute paths and failed after moving data.
   Fingerprints now use logical names, file size, and edge hashes.
3. The project declared EEGNet and MSCNN-CAM without an executable deep runner.
   Memory-mapped caches, training, validation, checkpointing, and prediction are
   now implemented and smoke-tested.
4. RTX setup did not force a CUDA PyTorch wheel or capture the active accelerator.
   The setup and experiment records now include CUDA, cuDNN, GPU identity, and
   package versions.
5. The data tree required PSG files after transfer. A verified alignment manifest
   now preserves official timing in the EEG-only archive.
6. Figures 6, 7, 10, and 11 differed materially from the paper's composition.
   They were regenerated after visual comparison with the PDF.
7. Nested deep validation failed for participants with too few independent
   class groups, and exclusions were silent. Split eligibility is now preflighted,
   exclusions are saved, and task-specific shared-cohort allowlists are available
   to both classical and deep runners.

## Retained limitations

- Fixed 30-second temporal bins are evaluation groups. They are not claimed to be
  the annotation file's unreliable auxiliary sequence column.
- Features named `abs_*` are absolute only when the preprocessing does not apply
  per-window z-score normalization. Run metadata records the normalization.
- Figure 11 is not an exact ICA reproduction because the paper does not publish
  the rejected components or rejection criteria.
- The local MSCNN-CAM is not the official Table 9 implementation.
