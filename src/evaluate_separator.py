import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.saved_synthetic_dataset import SavedSyntheticDataset, custom_collate_saved
from src.losses import permutation_invariant_l1_loss
from src.model_factory import build_model
from src.visualization_utils import match_predictions, save_separation_grid

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def alpha_bin(alpha):
    try:
        value = float(alpha)
    except (TypeError, ValueError):
        return "Unknown"
    if value < 0.4:
        return "0.30-0.40"
    if value < 0.5:
        return "0.40-0.50"
    if value < 0.6:
        return "0.50-0.60"
    if value <= 0.7:
        return "0.60-0.70"
    return "Other"


def write_grouped(results, key, output_path, transform=lambda x: x):
    grouped = {}
    for row in results:
        group = transform(row.get(key, ""))
        if group == "":
            group = "Unknown"
        entry = grouped.setdefault(group, {"count": 0, "l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0})
        entry["count"] += 1
        entry["l1"] += row["l1_loss"]
        entry["mse"] += row["mse"]
        entry["psnr"] += row["psnr"]
        if "ssim" in row:
            entry["ssim"] += row["ssim"]

    rows = []
    for group, values in grouped.items():
        count = values["count"]
        out = {
            key: group,
            "count": count,
            "l1_loss": values["l1"] / count,
            "mse": values["mse"] / count,
            "psnr": values["psnr"] / count,
        }
        if ssim is not None:
            out["ssim"] = values["ssim"] / count
        rows.append(out)

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a double-exposure separator.")
    parser.add_argument("--model", choices=["dual_head_unet", "two_decoder_unet", "two_stage_refinement_unet"], default="two_decoder_unet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SavedSyntheticDataset(args.manifest, image_size=args.image_size)
    if args.max_samples:
        dataset.rows = dataset.rows[:args.max_samples]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_saved)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", args.base_channels))
    model_name = checkpoint.get("model") or ckpt_args.get("model") or args.model
    model = build_model(model_name, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []
    visual_batch = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            mix = batch["mixture"].to(device)
            s1 = batch["source_1"].to(device)
            s2 = batch["source_2"].to(device)
            p1, p2 = model(mix)
            _, loss_a, loss_b, is_a = permutation_invariant_l1_loss(p1, p2, s1, s2)
            p1m, p2m = match_predictions(p1, p2, s1, s2, is_a)

            if visual_batch is None:
                visual_batch = (
                    s1.detach().cpu(),
                    s2.detach().cpu(),
                    mix.detach().cpu(),
                    p1m.detach().cpu(),
                    p2m.detach().cpu(),
                )

            for i in range(mix.size(0)):
                mse_1 = torch.mean((p1m[i] - s1[i]) ** 2).item()
                mse_2 = torch.mean((p2m[i] - s2[i]) ** 2).item()
                row = {
                    "sample_index": len(results),
                    "l1_loss": min(loss_a[i].item(), loss_b[i].item()),
                    "mse": (mse_1 + mse_2) / 2,
                    "psnr": (compute_psnr(p1m[i], s1[i]) + compute_psnr(p2m[i], s2[i])) / 2,
                }

                if ssim is not None:
                    p1_np = torch.clamp(p1m[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    p2_np = torch.clamp(p2m[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    s1_np = torch.clamp(s1[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    s2_np = torch.clamp(s2[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    row["ssim"] = float((ssim(p1_np, s1_np, data_range=1.0, channel_axis=-1) + ssim(p2_np, s2_np, data_range=1.0, channel_axis=-1)) / 2)

                metadata = batch["metadata"][i] if i < len(batch["metadata"]) else {}
                for key, value in metadata.items():
                    row.setdefault(key, value)
                results.append(row)

    if visual_batch is not None:
        save_separation_grid(args.output_dir / "eval_samples.png", *visual_batch)

    metrics = {
        "num_samples": len(results),
        "l1_loss": sum(r["l1_loss"] for r in results) / max(len(results), 1),
        "mse": sum(r["mse"] for r in results) / max(len(results), 1),
        "psnr": sum(r["psnr"] for r in results) / max(len(results), 1),
    }
    if ssim is not None and results and "ssim" in results[0]:
        metrics["ssim"] = sum(r["ssim"] for r in results) / len(results)

    with open(args.output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if results:
        with open(args.output_dir / "eval_per_sample.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    write_grouped(results, "synthesis_mode", args.output_dir / "eval_by_synthesis_mode.csv")
    write_grouped(results, "alpha", args.output_dir / "eval_by_alpha_bin.csv", transform=alpha_bin)
    write_grouped(results, "same_label", args.output_dir / "eval_by_same_label.csv")

    print(f"Evaluation complete: {metrics}")


if __name__ == "__main__":
    main()
