# Missing Presentation Assets

The following assets were **not found** in the current experiments directory.
They will be generated when you run a stronger training + evaluation pass.

## Missing Files

| Expected Path | Reason |
|---------------|--------|
| `experiments/baseline_unet/loss_curve.png` | The initial 5-epoch baseline run used an older script version that did not save loss curves. Re-run training with the updated `train_baseline.py` to generate this. |
| `experiments/baseline_unet/eval/eval_samples.png` | The evaluation script now saves `eval_samples.png`. Re-run `evaluate_baseline.py` against the new checkpoint to generate this. |
| `experiments/baseline_unet/eval/eval_by_alpha_bin.csv` | Alpha-bin grouping may not have had enough variety in the small validation set. Will be generated with a larger dataset. |

## Found and Copied

| Copied To | Original Path |
|-----------|---------------|
| `dataset_sample_check.png` | `results/dataset_sample_check.png` |
| `overfit_loss_curve.png` | `experiments/overfit_test/loss_curve.png` |
| `baseline_epoch_005_samples.png` | `experiments/baseline_unet/visuals/epoch_005_samples.png` |

## How to Regenerate

After running a full training pass:

```bash
# Train
python3 src/train_baseline.py \
  --train-manifest data/processed/synthetic/train/pair_manifest.csv \
  --val-manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean \
  --epochs 30 --batch-size 4 --lr 1e-4 --base-channels 32 --seed 585

# Evaluate
python3 src/evaluate_baseline.py \
  --checkpoint experiments/baseline_unet_linear_clean/checkpoints/best.pt \
  --manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean/eval \
  --batch-size 8

# Then copy new assets:
cp experiments/baseline_unet_linear_clean/loss_curve.png docs/presentation/assets/
cp experiments/baseline_unet_linear_clean/eval/eval_samples.png docs/presentation/assets/
```
