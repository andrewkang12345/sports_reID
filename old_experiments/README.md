# Archived experiments

This directory contains superseded experiments retained for provenance. None of
these files are required by `configs/best_stack.yaml`.

- `configs/`: v18-v55, MixSort, and standalone PARSeq trial configurations.
- `scripts/`: historical evaluators and post-processing experiments.
- `reid_benchmarks/`: CLIP, TransReID, SOLIDER, DINOv2, OSNet, and OpenGait sweeps.
- `mixsort/`: the archived local MixSort wrapper and vendored MixViT/MixSort source.
- `data/groundTruth_ARG_FRA_183303/`: the older partial ground-truth set used by
  archived OCR and frame evaluators.
- `data/README.roboflow.txt`: export metadata for that partial annotation set.

The full tracking ground truth and `eval_tracking.py` remain at repository root
because they validate the current best stack.

Archived BoxMOT and MixSort checkpoints can be restored separately:

```bash
python old_experiments/scripts/download_experiment_checkpoints.py
```

The local MixSort implementation is deliberately disconnected from the active
tracker factory. It is retained as source history, not as a supported runtime backend.
