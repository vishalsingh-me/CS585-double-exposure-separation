# Final Run Guide

Step-by-step commands to reproduce the full project pipeline.  
All commands assume you are in the **project root** directory.

---

## Prerequisites

```bash
source .venv/bin/activate
```

Make sure `torch`, `torchvision`, `Pillow`, `scikit-image`, `tqdm`, `matplotlib`, and `python-pptx` are installed.

---

## Step 1: Check Dataset Samples

Verify that the synthetic dataset loads correctly and tensors are valid.

```bash
python3 scripts/check_dataset_sample.py \
  --manifest data/processed/synthetic/train/pair_manifest.csv \
  --output results/dataset_sample_check.png \
  --num-samples 4
```

**Expected output:** Tensor stats printed, grid saved to `results/dataset_sample_check.png`.

---

## Step 2: Run Overfit Sanity Check

Train on a tiny subset to confirm the model can memorise.

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

**Expected output:**  
- Loss should decrease substantially (e.g. 0.467 → 0.174).  
- Results in `experiments/overfit_test/` (loss curve, sample grids, summary JSON).

---

## Step 3: Train Stronger Baseline

### Option A — CPU / MPS (moderate)

```bash
python3 src/train_baseline.py \
  --train-manifest data/processed/synthetic/train/pair_manifest.csv \
  --val-manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean \
  --epochs 30 \
  --batch-size 4 \
  --lr 1e-4 \
  --base-channels 32 \
  --seed 585
```

### Option B — GPU (stronger)

```bash
python3 src/train_baseline.py \
  --train-manifest data/processed/synthetic/train/pair_manifest.csv \
  --val-manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean_64ch \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --base-channels 64 \
  --seed 585
```

**Outputs (in `--output-dir`):**
- `checkpoints/best.pt` and `checkpoints/latest.pt`
- `loss_curve.png`
- `experiment_summary.json`
- `train_log.csv`
- `visuals/epoch_NNN_samples.png`

---

## Step 4: Evaluate

```bash
python3 src/evaluate_baseline.py \
  --checkpoint experiments/baseline_unet_linear_clean/checkpoints/best.pt \
  --manifest data/processed/synthetic/val/pair_manifest.csv \
  --output-dir experiments/baseline_unet_linear_clean/eval \
  --batch-size 8
```

**Outputs (in `--output-dir`):**
- `eval_metrics.json` — overall L1, PSNR, SSIM
- `eval_samples.png` — visual grid of predictions
- `eval_by_alpha_bin.csv` — metrics grouped by alpha range
- `eval_by_synthesis_mode.csv` — metrics grouped by blending mode
- `eval_by_same_label.csv` — metrics grouped by same/different label
- `eval_per_sample.csv` — per-sample detailed results

---

## Step 5: Update PPTX Manually

Open the presentation:

```
docs/presentation/CS585_Double_Exposure_Separation_Starter.pptx
```

Insert these files into the appropriate slides:

| Slide | Insert | Source File |
|-------|--------|-------------|
| 4 | Dataset examples | `results/dataset_sample_check.png` |
| 6 | Loss curve | `experiments/baseline_unet_linear_clean/loss_curve.png` |
| 6 | Prediction samples | `experiments/baseline_unet_linear_clean/visuals/epoch_030_samples.png` |
| 6 | Eval samples | `experiments/baseline_unet_linear_clean/eval/eval_samples.png` |
| 6 | Metric numbers | `experiments/baseline_unet_linear_clean/eval/eval_metrics.json` |
| 7 | Comparison grid | Any improved prediction grid from a stronger run |

Also update the numeric results on Slide 6:
- Read `eval_metrics.json` for L1, PSNR, SSIM values.
- Read `eval_by_alpha_bin.csv` for breakdowns by alpha.
- Read `eval_by_synthesis_mode.csv` for breakdowns by mode.

---

## Step 6: Regenerate Presentation (optional)

If you want to regenerate the PPTX template from scratch:

```bash
python3 scripts/generate_presentation.py
```

This overwrites `docs/presentation/CS585_Double_Exposure_Separation_Starter.pptx`.

---

## Notes

- All scripts auto-create output directories.
- All scripts support `--help` for argument documentation.
- Loss curves and experiment summaries are saved automatically.
- Visualisations clamp values to [0, 1].
