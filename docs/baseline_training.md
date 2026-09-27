# Baseline Training Guide

## Objective
The baseline training pipeline trains a dual-head U-Net model to take a single double-exposure mixture image and predict its two original source images. This separation step represents the inverse of the synthetic data generation pipeline.

## Architecture
The `DualHeadUNet` is an encoder-decoder architecture with skip connections. 
- **Input:** 3-channel RGB double-exposure image
- **Encoder:** Series of convolutional blocks and max pooling layers acting as feature extractors to downsample the image hierarchically.
- **Decoder:** Series of transpose convolutions or upsampling interpolations merging deep semantic features and shallow high-resolution visual features.
- **Output:** 6-channel feature map, which is sliced into two 3-channel representations (`pred_1` and `pred_2`), corresponding to the two separated sources.

## Permutation-Invariant Loss
The order in which the two source images are placed in the output channels is ambiguous. The model can output source A in head 1 and source B in head 2, or vice versa. Both are considered correct.
The `permutation_invariant_l1_loss` calculates the L1 reconstruction loss under both possible valid permutations and assigns the lowest loss permutation back to the model:
`final_loss = min( loss(pred_1, src_1) + loss(pred_2, src_2) , loss(pred_1, src_2) + loss(pred_2, src_1) )`

## Project Workflow

### Step 1: Generate synthetic train and val pairs
Generate a strong batch of samples for training:
```bash
python3 scripts/generate_synthetic_pairs.py \
  --input-csv data/processed/subsets/places365_small_splits/train.csv \
  --image-root data/interim/subsets/places365_small \
  --output-dir data/processed/synthetic \
  --split train \
  --num-pairs 5000 \
  --image-size 256 \
  --mode linear_clean \
  --alpha-min 0.3 \
  --alpha-max 0.7 \
  --seed 585

python3 scripts/generate_synthetic_pairs.py \
  --input-csv data/processed/subsets/places365_small_splits/val.csv \
  --image-root data/interim/subsets/places365_small \
  --output-dir data/processed/synthetic \
  --split val \
  --num-pairs 500 \
  --image-size 256 \
  --mode linear_clean \
  --alpha-min 0.3 \
  --alpha-max 0.7 \
  --seed 586
```

### Step 2: Check dataset samples
Verify the dataset images do not exhibit weird tinting or negative max/mins using the diagnostic script.
```bash
python3 scripts/check_dataset_sample.py \
  --manifest data/processed/synthetic/train/pair_manifest.csv \
  --output results/dataset_sample_check.png \
  --num-samples 4
```

### Step 3: Run overfit sanity check
Prior to starting large batches on your baseline network, evaluate its capacity to fit simple memory parameters!
```bash
python3 scripts/overfit_sanity_check.py \
  --manifest data/processed/synthetic/train/pair_manifest.csv \
  --output-dir experiments/overfit_test \
  --epochs 100 \
  --batch-size 2 \
  --base-channels 32 \
  --max-samples 16 \
  --lr 1e-4 \
  --seed 585
```
Check `experiments/overfit_test/loss_curve.png` to confirm the model reaches near-zero loss.

### Step 4: Train baseline U-Net
Launch a complete end-to-end training cycle on the stored 5000 images configurations:
```bash
python3 src/train_baseline.py \
  --train-manifest data/processed/synthetic/train/pair_manifest.csv \
  --val-manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --base-channels 64 \
  --seed 585
```
*(If you are memory constrained, set `--batch-size 4`, `--base-channels 32`, and `--image-size 128`)*

### Step 5: Evaluate baseline
Perform a rigorous qualitative breakdown on validation combinations tracking Alpha ranges, identical labels, and visual outputs:
```bash
python3 src/evaluate_baseline.py \
  --checkpoint experiments/baseline_unet_linear_clean/checkpoints/best.pt \
  --manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean/eval \
  --batch-size 8 \
  --image-size 256
```

### Step 6: Inspect visual results and grouped metrics
The pipeline populates robust results at `experiments/baseline_unet_linear_clean/`.
- Review structural data within `eval/eval_metrics.json`.
- Visualize the separated output via `eval/eval_samples.png`.
- Inspect any underlying dataset anomalies using `eval_by_alpha_bin.csv`.
