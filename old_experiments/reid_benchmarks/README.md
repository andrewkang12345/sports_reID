# Archived ReID experiments

`old_experiments/reid_benchmarks/benchmark_reid_models.py` compares ReID
representations on the 31 annotated frames in
`groundTruth_AllTracking_ARG_FRA_183303`.

The frame-level models use identical ground-truth crops and report:

- leave-one-frame-out rank-1, rank-5, and mAP;
- matching number-occluded `"for tracker"` crops to number-visible crops;
- one-to-one assignment accuracy between adjacent annotated frames; and
- sequence retrieval after splitting each player's observations into alternating
  temporal halves.

OpenGait is sequence-only. YOLO segmentation converts the annotated RGB boxes to
silhouettes, then the official GaitBase Gait3D checkpoint embeds each temporal half.
Its result is directly comparable only to the `temporal_split_retrieval` metric.

Example:

```bash
python old_experiments/reid_benchmarks/benchmark_reid_models.py \
  --model boxmot \
  --weights clip_market1501.pt \
  --device cuda:0 \
  --output old_experiments/reid_benchmarks/results/clip_market1501.json
```

To test an additional representation on top of CLIP:

```bash
python old_experiments/reid_benchmarks/benchmark_reid_models.py \
  --model fusion \
  --secondary transreid \
  --fusion-weights 0.25 0.5 0.75 \
  --device cuda:0 \
  --output old_experiments/reid_benchmarks/results/clip_plus_transreid.json
```

Fusion concatenates normalized embeddings with square-root weights, which is
equivalent to a weighted average of their cosine similarities. The selected weight
maximizes tracker-to-visible rank-1.

Use `--primary` to test a different base representation:

```bash
python old_experiments/reid_benchmarks/benchmark_reid_models.py \
  --model fusion \
  --primary osnet_ain \
  --secondary transreid \
  --device cuda:0 \
  --output old_experiments/reid_benchmarks/results/osnet_ain_plus_transreid.json
```

OpenGait fusion is sequence-level because gait requires multiple silhouettes:

```bash
python old_experiments/reid_benchmarks/benchmark_reid_models.py \
  --model fusion \
  --primary transreid \
  --secondary opengait \
  --device cuda:0 \
  --output old_experiments/reid_benchmarks/results/transreid_plus_opengait.json
```

For an end-to-end BoT-SORT run with TransReID replacing CLIP:

```bash
python run_demo.py \
  --video sample_data/test_clip.mp4 \
  --metadata sample_data/test_metadata.json \
  --output_dir outputs/best_stack \
  --config configs/best_stack.yaml
```

External official repositories and public checkpoints are expected under
`/mnt/data/reid_models` by default. The result files contain metrics and model
provenance only; no FIFA video frames or crops are written.

One-time setup for the external models:

```bash
pip install gdown yacs transformers huggingface_hub kornia

mkdir -p /mnt/data/reid_models/checkpoints
git clone https://github.com/damo-cv/TransReID.git /mnt/data/reid_models/TransReID
git clone https://github.com/tinyvision/SOLIDER-REID.git /mnt/data/reid_models/SOLIDER-REID
git clone https://github.com/ShiqiYu/OpenGait.git /mnt/data/reid_models/OpenGait

gdown 1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK \
  -O /mnt/data/reid_models/checkpoints/transreid_msmt17_vit.pth
gdown 1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b \
  -O /mnt/data/reid_models/checkpoints/solider_swin_small_msmt17.pth

python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "opengait/OpenGait",
    "Gait3D/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt",
    local_dir="/mnt/data/reid_models/checkpoints/opengait",
)
PY
```

## Official model sources

| Model | Code | Checkpoint used |
| --- | --- | --- |
| TransReID ViT-B MSMT17 | [damo-cv/TransReID](https://github.com/damo-cv/TransReID) | [Official MSMT17 model](https://drive.google.com/file/d/1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK/view) |
| SOLIDER Swin-S MSMT17 | [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID) | [Official MSMT17 model](https://drive.google.com/file/d/1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b/view) |
| DINOv2-small | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) |
| OpenGait GaitBase | [ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait) | [Official Gait3D checkpoint](https://huggingface.co/opengait/OpenGait/blob/main/Gait3D/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt) |

OpenGait's repository states that its code is limited to academic use. Review its
license before using that adapter outside research.

TF-CLIP was inspected but not scored. Its official pretrained MARS, LS-VID, and
iLIDS packages are exposed only through Baidu Pan browser pages, and no equivalent
checkpoint was found in a public package repository. The current environment could
open the password pages but could not obtain a direct model artifact.

The official [SoccerNet ReID kit](https://github.com/SoccerNet/sn-reid) provides a
`resnet50_fc512` training baseline and downloads the SoccerNet-v3 ReID dataset, but it
does not publish a SoccerNet-trained checkpoint. The OSNet MSMT17 and OSNet-AIN
experiments in this directory are therefore the executable strong general-purpose
baselines; they should not be described as SoccerNet-trained.

See [ARG_FRA_183303_RESULTS.md](ARG_FRA_183303_RESULTS.md) for the measured
comparison and end-to-end A/B result.
