import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import csv
import datetime
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.saved_synthetic_dataset import SavedSyntheticDataset, custom_collate_saved
from src.losses import get_reconstruction_loss, output_correlation_loss, permutation_invariant_l1_loss
from src.model_factory import build_model
from src.visualization_utils import match_predictions, save_separation_grid


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metadata_alphas(metadata, device):
    values = []
    for item in metadata:
        try:
            values.append(float(item.get("alpha", -1.0)))
        except (TypeError, ValueError):
            values.append(-1.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def plot_loss_curve(rows, output_path):
    if not rows:
        return
    width, height = 900, 560
    margin_l, margin_r, margin_t, margin_b = 70, 30, 60, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 24), "Separator Training Loss", fill=(20, 20, 20))

    series = [
        ("Train total", [row["train_total_loss"] for row in rows], (40, 95, 180)),
        ("Val total", [row["val_total_loss"] for row in rows], (210, 80, 60)),
        ("Train PIT", [row["train_pit_loss"] for row in rows], (40, 150, 90)),
        ("Val PIT", [row["val_pit_loss"] for row in rows], (140, 80, 170)),
    ]
    all_values = [value for _, values, _ in series for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    if y_max == y_min:
        y_max = y_min + 1.0

    def xy(index, value):
        x = margin_l + (index / max(len(rows) - 1, 1)) * plot_w
        y = margin_t + (1 - (value - y_min) / (y_max - y_min)) * plot_h
        return x, y

    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(180, 180, 180))
    for tick in range(5):
        y = margin_t + tick * plot_h / 4
        value = y_max - tick * (y_max - y_min) / 4
        draw.line((margin_l, y, margin_l + plot_w, y), fill=(235, 235, 235))
        draw.text((10, y - 7), f"{value:.3f}", fill=(80, 80, 80))

    for label, values, color in series:
        points = [xy(i, value) for i, value in enumerate(values)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=3)

    draw.text((margin_l + plot_w // 2 - 20, height - 40), "Epoch", fill=(40, 40, 40))
    legend_x = margin_l + plot_w - 160
    legend_y = 24
    for idx, (label, _, color) in enumerate(series):
        y = legend_y + idx * 18
        draw.line((legend_x, y + 8, legend_x + 24, y + 8), fill=color, width=3)
        draw.text((legend_x + 30, y), label, fill=(40, 40, 40))

    image.save(output_path)


def run_epoch(model, loader, device, optimizer, lambda_recon, lambda_corr, train):
    model.train(train)
    totals = {"total": 0.0, "pit": 0.0, "recon": 0.0, "corr": 0.0, "count": 0}
    first_visual_batch = None
    iterator = tqdm(loader, desc="Train" if train else "Val")

    for batch in iterator:
        mix = batch["mixture"].to(device)
        s1 = batch["source_1"].to(device)
        s2 = batch["source_2"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            p1, p2 = model(mix)
            pit_loss, _, _, is_a = permutation_invariant_l1_loss(p1, p2, s1, s2)
            recon_loss = torch.tensor(0.0, device=device)
            if lambda_recon > 0:
                recon_loss = get_reconstruction_loss(p1, p2, mix, metadata_alphas(batch["metadata"], device), is_a)
            corr_loss = output_correlation_loss(p1, p2)
            total_loss = pit_loss + lambda_recon * recon_loss + lambda_corr * corr_loss

            if train:
                total_loss.backward()
                optimizer.step()

        batch_size = mix.size(0)
        totals["total"] += total_loss.item() * batch_size
        totals["pit"] += pit_loss.item() * batch_size
        totals["recon"] += recon_loss.item() * batch_size
        totals["corr"] += corr_loss.item() * batch_size
        totals["count"] += batch_size
        iterator.set_postfix(total=f"{total_loss.item():.4f}", pit=f"{pit_loss.item():.4f}")

        if not train and first_visual_batch is None:
            p1m, p2m = match_predictions(p1, p2, s1, s2, is_a)
            first_visual_batch = (
                s1.detach().cpu(),
                s2.detach().cpu(),
                mix.detach().cpu(),
                p1m.detach().cpu(),
                p2m.detach().cpu(),
            )

    count = max(totals["count"], 1)
    metrics = {key: totals[key] / count for key in ["total", "pit", "recon", "corr"]}
    return metrics, first_visual_batch


def main():
    parser = argparse.ArgumentParser(description="Train a double-exposure separator.")
    parser.add_argument("--model", choices=["dual_head_unet", "two_decoder_unet", "two_stage_refinement_unet"], default="two_decoder_unet")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=585)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--lambda-corr", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--visualize-every", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    visual_dir = args.output_dir / "visuals"
    checkpoint_dir.mkdir(exist_ok=True)
    visual_dir.mkdir(exist_ok=True)

    train_ds = SavedSyntheticDataset(args.train_manifest, image_size=args.image_size)
    val_ds = SavedSyntheticDataset(args.val_manifest, image_size=args.image_size)
    if args.max_train_samples:
        train_ds.rows = train_ds.rows[:args.max_train_samples]
    if args.max_val_samples:
        val_ds.rows = val_ds.rows[:args.max_val_samples]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_saved,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_saved,
    )

    model = build_model(args.model, base_channels=args.base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fields = [
        "epoch",
        "train_total_loss",
        "train_pit_loss",
        "train_recon_loss",
        "train_corr_loss",
        "val_total_loss",
        "val_pit_loss",
        "val_recon_loss",
        "val_corr_loss",
        "learning_rate",
    ]
    rows = []
    best_val = float("inf")
    with open(args.output_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics, _ = run_epoch(model, train_loader, device, optimizer, args.lambda_recon, args.lambda_corr, train=True)
            with torch.no_grad():
                val_metrics, visual_batch = run_epoch(model, val_loader, device, optimizer, args.lambda_recon, args.lambda_corr, train=False)

            row = {
                "epoch": epoch,
                "train_total_loss": train_metrics["total"],
                "train_pit_loss": train_metrics["pit"],
                "train_recon_loss": train_metrics["recon"],
                "train_corr_loss": train_metrics["corr"],
                "val_total_loss": val_metrics["total"],
                "val_pit_loss": val_metrics["pit"],
                "val_recon_loss": val_metrics["recon"],
                "val_corr_loss": val_metrics["corr"],
                "learning_rate": args.lr,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()

            if visual_batch is not None and epoch % args.visualize_every == 0:
                save_separation_grid(visual_dir / f"epoch_{epoch:03d}_samples.png", *visual_batch)

            checkpoint = {
                "epoch": epoch,
                "model": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_total_loss": val_metrics["total"],
                "val_pit_loss": val_metrics["pit"],
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }
            torch.save(checkpoint, checkpoint_dir / "latest.pt")
            if epoch % args.save_every == 0:
                torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                torch.save(checkpoint, checkpoint_dir / "best.pt")

            plot_loss_curve(rows, args.output_dir / "loss_curve.png")
            print(
                f"Epoch {epoch}: train total={train_metrics['total']:.4f}, "
                f"val total={val_metrics['total']:.4f}, val PIT={val_metrics['pit']:.4f}"
            )

    summary = {
        "model": args.model,
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "lambda_recon": args.lambda_recon,
        "lambda_corr": args.lambda_corr,
        "best_val_total_loss": best_val,
        "final": rows[-1] if rows else {},
        "device": str(device),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "datetime": str(datetime.datetime.now()),
    }
    with open(args.output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
