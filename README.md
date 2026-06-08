# Soccer Player Identity Prototype

This repository is a runnable prototype for roster-conditioned soccer player identity detection in short broadcast clips. It predicts identity at the tracklet level, extracts players close to the ball for at least a configured duration, and renders an annotated video.

The default path is dependency-light and uses OpenCV fallbacks so it runs without downloaded model weights. The code is structured around adapter interfaces where YOLO/RT-DETR/SAM/SAM2/Sapiens, ByteTrack/BoT-SORT, SoccerNet ReID, jersey-number pipelines, and face/headshot models can be plugged in.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

# Install the PyTorch build appropriate for your CPU/CUDA environment first.
pip install torch torchvision
pip install opencv-python numpy pyyaml scipy pillow imageio-ffmpeg ultralytics boxmot==10.0.83 gdown
```

The repository does not store model binaries. Restore the exact checkpoint set used by
the checked-in configs (about 1.7 GB) from the public upstream projects:

```bash
python scripts/download_checkpoints.py
```

The downloader skips files that are already present and verifies every file against the
SHA-256 digest of the development checkout. Use `--group yolo` or
`--group reid-ocr` for a partial download.

### Public Checkpoint Sources

| Local path(s) | Public source |
| --- | --- |
| `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11m-seg.pt`, `yolo11x-seg.pt`, `yolo11m-pose.pt` | [Ultralytics](https://github.com/ultralytics/ultralytics) release assets, downloaded by the `ultralytics` package |
| `clip_market1501.pt`, `osnet_x0_25_msmt17.pt`, `osnet_x1_0_msmt17.pt`, `osnet_ain_x1_0_msmt17.pt` | [BoxMOT ReID model zoo](https://github.com/mikel-brostrom/boxmot) |
| `models/parseq_soccernet.ckpt`, `models/mkoshkina_legibility_resnet34.pth` | [A General Framework for Jersey Number Recognition in Sports Video](https://github.com/mkoshkina/jersey-number-pipeline) |
| `models/mixsort/MixFormer_soccernet_train.pth.tar` | [MixSort model zoo](https://github.com/MCG-NJU/MixSort) |

The upstream projects control the checkpoint licenses and usage terms. Review those
terms before redistribution or commercial use.

## Metadata

`metadata.json` must contain teams and roster entries:

```json
{
  "home_team": "Molde FK",
  "away_team": "Rosenborg BK",
  "team_colors": {
    "Molde FK": "#d22323",
    "Rosenborg BK": "#1e4bd7"
  },
  "rosters": {
    "Molde FK": [
      {
        "player_name": "Ola Brynhildsen",
        "jersey_number": 9,
        "position": "FW",
        "headshot_path": "assets/headshots/molde/ola.jpg"
      }
    ]
  }
}
```

`team_colors` is optional but useful for the fallback team classifier. `headshot_path` and `headshot_url` are optional; local `headshot_path` values are resolved relative to the metadata file.

## Run Inference

```bash
python run_demo.py \
  --video path/to/your_clip.mp4 \
  --metadata sample_data/test_metadata.json \
  --output_dir outputs/test_demo \
  --config configs/default.yaml
```

Outputs:

```text
outputs/test_demo/
  result.json
  visualization.mp4
  debug_tracks.json
  debug_identity_scores.json
  debug_events.csv
```

`result.json` follows the requested schema with `clip_path`, `home_team`, `away_team`, and `events`.
`visualization.mp4` is transcoded to H.264 `avc1` with `yuv420p` pixels by default so it previews in VS Code and browsers. Disable this with `visualization.browser_compatible_mp4: false`.

## Video Data

No video files are included in this repository. The development clips and rendered
visualizations are ignored by Git because the broadcast source used during development
comes from the gated FIFA Skeletal Tracking Light 2026 dataset.

Developers who independently have access to that dataset can prepare the same kind of
10-second H.264 clip from their local copy:

```bash
python sample_data/prepare_broadcast_clip.py
```

This writes:

```text
sample_data/test_clip.mp4
sample_data/test_metadata.json
```

The script currently expects `BRA_KOR_230503.mp4` from FIFA Skeletal Tracking Light
2026 to be staged under `/mnt/data/mywork/fifaInnovationChallenge`; adjust its input
arguments for your local dataset location. The dataset page lists `cc-by-2.0` and gated
non-commercial access terms. The clip does not include real roster/headshot metadata, so
`test_metadata.json` marks `identity_labels_available: false` and uses anonymous roster
slots. The demo therefore renders conservative `Unknown` labels rather than pretending
to know real player names.

An additional Wikimedia Commons public-domain sideline clip downloader is available for
smoke tests:

```bash
python sample_data/download_public_clip.py
```

A synthetic fallback generator remains available for controlled regression testing:

```bash
python sample_data/generate_sample_data.py
```

## Pipeline

The main stages are:

1. Video ingestion: `soccer_identity/utils/video_io.py`
2. Player detection: `soccer_identity/detection/player_detector.py`
3. Ball detection: `soccer_identity/detection/ball_detector.py`
4. Tracking: `soccer_identity/tracking/tracker.py`
5. Tracklet generation: `soccer_identity/tracking/tracklet.py`
6. Team classification: `soccer_identity/identity/team_classifier.py`
7. Jersey OCR: `soccer_identity/identity/jersey_ocr.py`
8. Headshot matching: `soccer_identity/identity/headshot_matcher.py`
9. Body ReID: `soccer_identity/identity/body_reid.py`
10. Position prior: `soccer_identity/identity/position_prior.py`
11. Roster-conditioned fusion: `soccer_identity/identity/fusion.py`
12. Temporal/global assignment: `soccer_identity/identity/assignment.py`
13. Ball-proximity events: `soccer_identity/proximity/ball_proximity.py`
14. Visualization: `soccer_identity/visualization/render.py`

## Active Demo Backends

The default broadcast demo now uses:

- Player detector: Ultralytics YOLO11n COCO `person` class on CUDA when available.
- Ball detector: Ultralytics YOLO11n COCO `sports ball` class on CUDA, with OpenCV ball fallback.
- Tracker: Ultralytics ByteTrack IDs exposed through the detector, with simple IoU fallback if detector IDs are absent.
- Team/player gating: metadata kit colors, including shirt and shorts when available, plus sideline truncation checks to suppress referees, coaches, and staff where possible.
- Body appearance: local HSV histogram embedding, still a lightweight fallback rather than SoccerNet ReID.
- Jersey number: conservative template OCR fallback; real SoccerNet jersey-number weights are not bundled.
- Face/headshot: inactive unless real headshot paths are provided in metadata.

Low-confidence identities are displayed as `Low conf ID` in the visualization instead of implying a real player identity. Sapiens, SAM/SAM2, SoccerNet ReID, SoccerNet Jersey Number, and SportsMOT/MixSort are explicit adapter points, but they require their external repos/weights and are not silently used by the default demo.

## Replacing Perception Modules

Each module has a small interface:

- Replace `PlayerDetector.detect()` with YOLO, RT-DETR, Detectron2, Sapiens, or a SoccerNet detector.
- Replace `BallDetector.detect()` with a ball-specific detector.
- Replace `PlayerSegmenter.segment()` with SAM/SAM2/Sapiens masks.
- Replace `MultiObjectTracker.update()` with ByteTrack, BoT-SORT, OC-SORT, StrongSORT, DeepSORT, TrackLab, or SportsMOT-style tracking.
- Replace `JerseyOCR.recognize()` with SoccerNet jersey-number code, PaddleOCR, EasyOCR, or a custom jersey recognizer.
- Replace `extract_body_embedding()` with SoccerNet ReID or another person/sports ReID model.
- Replace `extract_head_embedding()` and `HeadshotMatcher` with ArcFace/InsightFace/CLIP-style face or head matching.

The identity resolver is roster-conditioned. It scores each tracklet against the current roster rather than using a fixed identity classifier, so it supports unseen teams and players at inference time.

## Production Training

The repo includes a production-oriented config:

```bash
python run_demo.py \
  --video path/to/clip.mp4 \
  --metadata path/to/metadata.json \
  --output_dir outputs/prod_demo \
  --config configs/production.yaml
```

`configs/production.yaml` expects external weights/checkouts for the detector and SportsMOT/MixSort-style tracker. For local smoke tests, keep `configs/default.yaml`.

Validate dataset roots and create a training manifest:

```bash
python prepare_production_training.py \
  --sportsmot_root /data/SportsMOT \
  --soccernet_gsr_root /data/SoccerNet-GSR \
  --soccernet_reid_root /data/SoccerNet-ReID \
  --soccernet_jersey_root /data/SoccerNet-Jersey \
  --soccertrack_root /data/SoccerTrack-v2 \
  --output_manifest outputs/production_manifest.json
```

SportsMOT is supported as a MOTChallenge-format dataset parser in `soccer_identity/training/dataset_adapters.py` and as an explicit tracker backend boundary in `soccer_identity/tracking/sportsmot_adapter.py`. The adapter falls back only in the default config; production config requires the external repo and weights to avoid silently pretending a production tracker is active.

## Train Fusion Model

The non-learned fusion baseline is used by default. A lightweight MLP trainer is provided for candidate-level feature rows:

```bash
python train_identity_fusion.py \
  --train_jsonl path/to/train_candidates.jsonl \
  --output_path outputs/fusion_head.pt
```

Each JSONL row should contain:

```json
{"features": [0.9, 0.2, 0.0, 0.7], "label": 1, "track_id": "12", "candidate_player_id": "team|9|name"}
```

For a production training set, build rows from frozen tracklet evidence: team logits, jersey logits, headshot similarity, body embedding similarity, position trajectory features, ball-relative trajectory features, crop/OCR/head quality, duration, and occlusion stats.

## Evaluation Hooks

`soccer_identity/training/evaluate.py` includes a near-ball interval F1 helper. Add dataset-specific adapters for:

- player detection mAP
- ball precision/recall
- tracking IDF1/HOTA/MOTA
- jersey-number accuracy
- team classification accuracy
- identity accuracy per tracklet
- timestamp boundary error
- ablations for tracking, team, jersey, headshot, body ReID, and full fusion

## Limitations

- The default OpenCV detector/tracker is only a runnable fallback. It is not a substitute for YOLO/RT-DETR plus ByteTrack/BoT-SORT on real broadcasts.
- The template jersey OCR fallback only handles clear high-contrast digits and intentionally suppresses weak evidence.
- Headshot matching uses a simple image embedding unless replaced with a face/head model.
- Body ReID uses color/appearance histograms by default; replace it with SoccerNet ReID or a sports/person ReID model for real performance.
- Position priors are weak unless pitch calibration is available.
- Image-space ball proximity is used when calibration is absent, so camera zoom and perspective can affect thresholds.

The prototype favors correctness over naming every player: low-confidence identity predictions are shown as `Unknown` or `Unknown #N` rather than hallucinated names.
