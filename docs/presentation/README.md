# CS585 Double-Exposure Separation — Presentation

## File Location

The starter PPTX deck is saved at:

```
docs/presentation/CS585_Double_Exposure_Separation_Starter.pptx
```

## Slide Overview

| Slide | Title | Notes |
|-------|-------|-------|
| 1 | Title Slide | Replace `[Name]` and `[Date]` placeholders |
| 2 | Problem Motivation | Insert a real double-exposure example image on the right |
| 3 | Data Generation Pipeline | Text-only; no images needed |
| 4 | Synthetic Dataset Examples | **Replace all 6 placeholder boxes** with crops from `results/dataset_sample_check.png` |
| 5 | Baseline Model | Insert a U-Net architecture diagram on the right placeholder |
| 6 | Preliminary Results & Findings | Insert `epoch_005_samples.png`, `loss_curve.png`, or `eval_samples.png` |
| 7 | Next Steps & Conclusion | Optionally insert comparison grid from a stronger training run |

## Slides That Need Manual Image Replacement

- **Slide 4** — Synthetic Dataset Examples (6 image placeholders)
- **Slide 5** — Model architecture diagram placeholder (right side)
- **Slide 6** — Results visuals (2 image placeholders)
- **Slide 7** — Comparison grid placeholder (right side)

## Assets to Insert

Copy these from the project into the slides:

| Asset | Source Path |
|-------|------------|
| Dataset sample grid | `results/dataset_sample_check.png` |
| Overfit loss curve | `experiments/overfit_test/loss_curve.png` |
| Baseline epoch samples | `experiments/baseline_unet/visuals/epoch_005_samples.png` |
| Baseline eval samples | `experiments/baseline_unet/eval/eval_samples.png` (if available) |
| Baseline loss curve | `experiments/baseline_unet/loss_curve.png` (if available) |

Pre-collected copies are in `docs/presentation/assets/`.

## How to Regenerate the PPTX

```bash
source .venv/bin/activate
python3 scripts/generate_presentation.py
```

The script is at `scripts/generate_presentation.py`.  
It requires `python-pptx` (`pip install python-pptx`).

## Which Experiment Outputs to Insert Later

After running a stronger training (e.g. 30 epochs, base_channels=32 or 64):

- `experiments/<run>/loss_curve.png` — training + validation loss curve
- `experiments/<run>/visuals/epoch_030_samples.png` — final epoch sample grid
- `experiments/<run>/eval/eval_samples.png` — evaluation visual grid
- `experiments/<run>/eval/eval_metrics.json` — L1, PSNR, SSIM summary
- `experiments/<run>/eval/eval_by_alpha_bin.csv` — metrics grouped by alpha
- `experiments/<run>/eval/eval_by_mode.csv` — metrics grouped by synthesis mode
