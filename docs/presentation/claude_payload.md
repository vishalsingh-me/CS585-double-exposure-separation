Experiment summary and artifacts to include in the slide.

Overwrite this file into Claude's working notes. It contains the exact numbers and paths to images for slide 6.

---

Overfit experiment (experiments/overfit_test):
- Path: experiments/overfit_test
- Summary file: experiments/overfit_test/experiment_summary.json
  - best_val_loss: 0.12956197187304497
  - final_train_loss: 0.1741769015789032
  - final_val_loss: 0.14591832272708416
  - epochs: 100
  - base_channels: 32
  - image_size: 256
  - num_train_samples: 16
  - device: mps
- Loss curve: experiments/overfit_test/loss_curve.png
- Per-epoch CSV log: experiments/overfit_test/train_log.csv
- Visuals (per-epoch sample grids): experiments/overfit_test/visuals/
  - Representative files: epoch_001_samples.png, epoch_050_samples.png, epoch_100_samples.png

Quick baseline experiment (experiments/baseline_unet):
- Path: experiments/baseline_unet
- Per-epoch CSV log: experiments/baseline_unet/train_log.csv (see epoch 5 for quick run metrics)
  - epoch 5 train_loss: 0.42980955123901365
  - epoch 5 val_loss: 0.42055012583732604
- Eval metrics: experiments/baseline_unet/eval/eval_metrics.json
  - psnr: 12.238126865451386
  - ssim: 0.28589877635240557
- Visuals folder: experiments/baseline_unet/visuals/
  - Use any sample grid image for predictions

Notes for Claude:
- The slide rounds numbers to 2 or 3 significant digits (e.g., 0.130 for best val L1, 12.24 dB PSNR). If you prefer precise values, use the raw numbers above.
- Files to embed in slide 6:
  - experiments/overfit_test/loss_curve.png (loss plot)
  - experiments/overfit_test/visuals/epoch_100_samples.png (prediction grid at final epoch)
  - experiments/baseline_unet/eval/ (sample prediction grid and eval_metrics.json for PSNR/SSIM values)

If you want me to insert these images into the PPTX directly, tell me which exact files to place and on which slides.