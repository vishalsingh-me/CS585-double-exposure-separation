# Project Status Summary

## 1. Project Goal

Separate a single double-exposure image into its two hidden source images.  
Given one blended photograph, the model predicts two output images that correspond to the original sources.

## 2. Why Synthetic Data Is Used

Real double-exposure photographs do not come with ground-truth source layers — there is no way to know what the two original images looked like. To train a supervised model, we generate synthetic double-exposure mixtures from clean images, which gives us perfect target labels.

## 3. Data Generation Pipeline

- **Source images:** Places365 scene dataset (train / val splits).
- **Sampling:** Two images are randomly paired per sample.
- **Blending modes:**
  - Linear alpha blending (`α · img₁ + (1−α) · img₂`)
  - Optional gradient mask blending
  - Optional Gaussian blur on one source
  - Optional gamma correction on the mixture
  - Optional spatial translation
  - Optional additive Gaussian noise
- **Saved outputs per pair:**
  - `mixture.png` (model input)
  - `source_1.png`, `source_2.png` (target labels)
  - Rich metadata in `pair_manifest.csv` (alpha, mode, blur radius, gamma, etc.)

## 4. Model Architecture

**Dual-Head U-Net** — a standard U-Net with 4 encoder and 4 decoder stages.

- **Input:** 1 mixture image (3 × 256 × 256)
- **Encoder:** Shared convolutional encoder with max-pooling (channels: C → 2C → 4C → 8C → 16C)
- **Decoder:** Shared decoder with transposed convolutions and skip connections
- **Output:** 6-channel final layer → split into `pred_1` (3 channels) and `pred_2` (3 channels)
- **Activation:** Sigmoid (outputs in [0, 1])

## 5. Loss Function

**Permutation-Invariant L1 Loss**

Because the two sources are unordered (the model's output 1 could correspond to either source), we compute both possible assignments and take the minimum:

```
loss_a = L1(pred_1, source_1) + L1(pred_2, source_2)
loss_b = L1(pred_1, source_2) + L1(pred_2, source_1)
final_loss = min(loss_a, loss_b)   (element-wise per batch, then averaged)
```

An optional reconstruction-consistency loss enforces that the predicted sources blend back to the mixture:
```
recon_loss = L1(α · pred_matched_1 + (1−α) · pred_matched_2, mixture)
```

## 6. Diagnostics Completed

| Check | Status | Notes |
|-------|--------|-------|
| Dataset sample check | ✅ Passed | Tensors are RGB, float32, bounded in [0, 1] |
| Overfit sanity check | ✅ Passed | Loss decreased from **0.467 → 0.174** over 100 epochs on 16 samples (base_channels=32) |
| End-to-end training | ✅ Working | 5-epoch baseline run completed successfully |
| Evaluation pipeline | ✅ Working | L1, PSNR, SSIM, and grouped metrics computed |

## 7. Current Findings

### Preliminary Baseline Run (5 epochs, base_channels=16)

| Metric | Value |
|--------|-------|
| Train L1 Loss | 0.430 |
| Val L1 Loss | 0.421 |
| PSNR | 12.24 dB |
| SSIM | 0.286 |

- The model learns a reasonable average but does not yet produce sharp separations.
- Visual predictions show the model is starting to separate colours and structure.
- Loss decreases consistently from epoch 1 to epoch 5.

### Overfit Sanity Check

- 100 epochs on 16 samples with `base_channels=32`
- Loss: **0.467 → 0.174** (train), best val: **0.130**
- This confirms the model, loss function, and data loading pipeline are all functional.

## 8. Limitations

- **Small training set:** The current dataset has limited diversity (few source images, one blending mode).
- **Weak model capacity:** `base_channels=16` was used for the initial run; a production baseline needs 32–64.
- **Short training:** Only 5 epochs were run; convergence requires 30+.
- **Linear-only blending:** The current results only evaluate the simplest blending mode (clean linear alpha). Harder modes (blur, mask, gamma, transform) have not yet been tested at scale.
- **No perceptual loss:** Only pixel-level L1 is used; perceptual or adversarial losses may improve sharpness.

## 9. Next Steps

1. **Scale training data** to 1,000–5,000 synthetic pairs.
2. **Increase model capacity** to `base_channels=32` or `64`.
3. **Train for 30+ epochs** to allow convergence.
4. **Evaluate by alpha range** (near 0.5 = hardest) and **synthesis mode** (blur, mask, etc.).
5. **Add reconstruction-consistency loss** (`lambda_recon > 0`) to enforce physical plausibility.
6. **Compare harder blending regimes:** blur, gradient mask, gamma correction, spatial shifts.
7. **Interpretability analysis:** study which factors (alpha, scene similarity, blur) make separation most difficult.

---

*This project builds a controlled synthetic framework for double-exposure separation and analyses which visual factors make separation difficult.*
