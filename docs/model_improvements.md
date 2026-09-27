# Model Improvements

## Two-Stage Residual Refinement U-Net (`two_stage_refinement_unet`)

**Why this model was added**: 
During initial experiments with the Single and Dual-Head U-Net architectures, we observed instances of ghosting artifacts and source leakage, where features of one source image were still visible in the separated output for the other. This model is introduced to specifically address these artifacts.

**Architecture Summary**:
The model utilizes a two-stage approach:
1. **Stage 1 (Coarse Separation)**: A shared-encoder U-Net with two independent decoder heads (identical to `TwoDecoderUNet`). This stage outputs two coarse prediction images (`coarse_1` and `coarse_2`).
2. **Stage 2 (Residual Refinement)**: A lightweight secondary U-Net (Refinement U-Net) takes the concatenated mixture and both coarse predictions (9 channels total) as input. It predicts a 6-channel residual tensor. The residuals are added to the coarse predictions and clamped to `[0, 1]`.

**Expected Benefit**:
By providing the refiner with both the original mixture and the initial coarse guesses, the network has full context to learn high-frequency corrections. It specifically targets removing residual ghosting/source leakage without needing an overly complex or heavy architecture.

**Training and Evaluation Commands**:

To train the model (using PIT-only loss for comparability):
```bash
python3 src/train_separator.py \
  --model two_stage_refinement_unet \
  --train-manifest data/processed/synthetic/train/pair_manifest.csv \
  --val-manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/two_stage_refinement_pit \
  --epochs 40 \
  --batch-size 4 \
  --lr 1e-4 \
  --base-channels 32 \
  --image-size 256 \
  --lambda-recon 0.0 \
  --lambda-corr 0.0 \
  --seed 585
```

To evaluate the model:
```bash
python3 src/evaluate_separator.py \
  --model two_stage_refinement_unet \
  --checkpoint experiments/two_stage_refinement_pit/checkpoints/best.pt \
  --manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/two_stage_refinement_pit/eval \
  --batch-size 4 \
  --base-channels 32 \
  --image-size 256
```
