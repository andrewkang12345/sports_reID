# Current best experiment

`configs/best_stack.yaml` is the best measured end-to-end configuration on
`groundTruth_AllTracking_ARG_FRA_183303`.

```bash
python run_demo.py \
  --video sample_data/test_clip.mp4 \
  --metadata sample_data/test_metadata.json \
  --output_dir outputs/best_stack \
  --config configs/best_stack.yaml

python eval_tracking.py outputs/best_stack
```

Measured result:

| Detection recall | Jersey accuracy | Identity accuracy | ID switches |
| ---: | ---: | ---: | ---: |
| 94.6% | 72.2% | 89.3% | 38 |

The stack uses TransReID ViT-B MSMT17 inside BoT-SORT and OSNet x1.0 MSMT17 for
cross-track appearance memory. The aggregate result is stored in
[`results/best_stack_eval.json`](results/best_stack_eval.json).

The TransReID repository and checkpoint are public:

```bash
mkdir -p /mnt/data/reid_models/checkpoints
git clone https://github.com/damo-cv/TransReID.git /mnt/data/reid_models/TransReID
gdown 1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK \
  -O /mnt/data/reid_models/checkpoints/transreid_msmt17_vit.pth
```

Previous configurations, comparison results, and benchmark tooling are archived in
[`old_experiments/`](../old_experiments/README.md).
