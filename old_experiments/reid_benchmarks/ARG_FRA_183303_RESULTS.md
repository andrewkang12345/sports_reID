# ARG-FRA 183303 ReID benchmark

Evaluated on all 521 player annotations in
`groundTruth_AllTracking_ARG_FRA_183303`: 115 number-visible crops and 406
number-occluded `"for tracker"` crops across 21 identities.

## Controlled embedding benchmark

| Model | All R1 | All mAP | Tracker-to-visible R1 | Adjacent assignment | Temporal R1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP + TransReID (0.35) | 79.27% | **46.68%** | **51.52%** | 84.84% | 100.00% |
| CLIP + OSNet-AIN (0.25) | 79.08% | 44.46% | 51.22% | 84.84% | 100.00% |
| OSNet-AIN + TransReID (0.40) | 79.08% | 43.34% | 46.65% | 81.35% | 100.00% |
| CLIP + SOLIDER (0.50) | 76.78% | 43.11% | 51.22% | 82.38% | 100.00% |
| CLIP + DINOv2 (0.10) | 76.58% | 41.76% | 48.17% | 80.53% | 100.00% |
| TransReID ViT-B MSMT17 | 75.62% | 44.05% | 39.33% | 80.74% | 100.00% |
| Current CLIP Market1501 | 74.28% | 42.36% | 48.48% | 81.35% | 100.00% |
| OSNet-AIN x1.0 MSMT17 | 69.48% | 34.90% | 41.46% | 69.26% | 95.00% |
| SOLIDER Swin-S MSMT17 | 66.22% | 34.53% | 39.63% | 76.02% | 100.00% |
| OSNet x0.25 MSMT17 | 56.62% | 27.82% | 26.52% | 65.16% | 95.00% |
| OSNet x1.0 MSMT17 | 55.09% | 25.97% | 19.21% | 63.93% | 95.00% |
| HSV histogram | 53.93% | 24.32% | 30.49% | 59.63% | 100.00% |
| DINOv2-small | 39.54% | 17.72% | 20.43% | 42.42% | 90.00% |
| OpenGait GaitBase Gait3D | - | - | - | - | 30.00% |

The appearance-model temporal split is saturated and is not useful for selecting a
winner. Tracker-to-visible retrieval and adjacent assignment are the relevant columns.
OpenGait is sequence-only; YOLO found silhouettes for 519/521 crops, but one-second
sampling and non-periodic soccer motion are a poor match for gait recognition.

Relative to current CLIP, the best controlled addition is TransReID at 0.35 weight:

- tracker-to-visible rank-1: +3.05 percentage points;
- all-crop rank-1: +4.99 points;
- all-crop mAP: +4.33 points; and
- adjacent assignment: +3.48 points.

CLIP + OSNet-AIN is within 0.30 points on tracker-to-visible rank-1 with a much
smaller checkpoint and feature vector, making it the practical candidate.

## OpenGait fusion

OpenGait produces one embedding per temporal silhouette sequence, so these pairings
can only be compared on temporal split retrieval. They cannot produce the frame-level
tracker-to-visible or adjacent-assignment metrics above.

| Appearance + OpenGait | Selected gait weight | Temporal R1 | Temporal mAP | Result |
| --- | ---: | ---: | ---: | --- |
| CLIP + OpenGait | 0.10 | 100.00% | 100.00% | No gain; degrades above 0.20 |
| TransReID + OpenGait | 0.10 | 100.00% | 100.00% | No gain; degrades above 0.60 |
| OSNet-AIN + OpenGait | 0.10 | 95.00% | 97.50% | No gain; degrades above 0.30 |

The appearance-only CLIP and TransReID temporal tests were already saturated at
100%. OpenGait therefore adds no measurable benefit on this clip. Its standalone
30% rank-1 result and degradation at high fusion weights make it unsuitable for the
current short, non-periodic soccer sequences.

## End-to-end A/B

`old_experiments/configs/v55_osnet_ain_sticky_stitch.yaml` changes only the
cross-track memory checkpoint
from OSNet x1.0 to OSNet-AIN x1.0. Detection, CLIP BoT-SORT association, OCR,
thresholds, and stitching remain fixed.

The current `configs/best_stack.yaml` instead starts from archived `v54` and changes
only BoT-SORT's
crop-level association encoder from CLIP Market1501 to the official TransReID ViT-B
MSMT17 checkpoint. Cross-track appearance memory remains OSNet x1.0.

| Run | Detection recall | Jersey accuracy | Identity accuracy | ID switches |
| --- | ---: | ---: | ---: | ---: |
| `v54` CLIP + OSNet x1.0 | 94.6% | **72.2%** | 88.1% | 39 |
| `v55` CLIP + OSNet-AIN x1.0 | 94.6% | 63.5% | 88.1% | 39 |
| `v56` TransReID + OSNet x1.0 | 94.6% | **72.2%** | **89.3%** | **38** |

Replacing the memory checkpoint with OSNet-AIN does not improve identity and makes
jersey overrides worse. Replacing CLIP inside BoT-SORT with TransReID does improve
the end-to-end result by 1.2 identity-accuracy points and one ID switch without
changing detection or jersey accuracy. `v56` is therefore the best measured
full-stack configuration on this clip, with the tradeoff of a much larger and slower
tracker embedding model.

## Unscored requested models

- **TF-CLIP:** official pretrained packages are available only through Baidu Pan
  browser pages. The password pages opened, but no direct checkpoint artifact was
  obtainable in this environment and no equivalent public package was found.
- **SoccerNet-ReID trained baseline:** the official kit provides training code and
  data download support, but no SoccerNet-trained checkpoint. Its baseline is
  `resnet50_fc512`, trained from scratch. The OSNet variants above are strong
  general-purpose baselines, not SoccerNet-trained models.

Raw metrics and fusion sweeps are in `old_experiments/reid_benchmarks/results/`.
