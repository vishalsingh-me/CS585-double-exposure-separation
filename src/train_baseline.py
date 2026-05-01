import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import csv
from tqdm import tqdm

from src.models.dual_head_unet import DualHeadUNet
from src.losses import permutation_invariant_l1_loss, get_reconstruction_loss, get_perceptual_loss, VGGLoss
from src.data.saved_synthetic_dataset import SavedSyntheticDataset, custom_collate_saved
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import datetime
import json

def create_grid(mixture, s1, s2, p1, p2, e1, e2):
    bz = mixture.shape[0]
    out = []
    for i in range(bz):
        # clamp to [0, 1] to avoid matplotlib/vision warnings and colored tinting
        out.extend([
            torch.clamp(s1[i], 0, 1), 
            torch.clamp(s2[i], 0, 1), 
            torch.clamp(mixture[i], 0, 1), 
            torch.clamp(p1[i], 0, 1), 
            torch.clamp(p2[i], 0, 1), 
            torch.clamp(e1[i], 0, 1), 
            torch.clamp(e2[i], 0, 1)
        ])
    return vutils.make_grid(out, nrow=7, normalize=False)

def plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Permutation-Invariant L1 Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Train a Dual-Head U-Net for double-exposure separation.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Path to training pair_manifest.csv")
    parser.add_argument("--val-manifest", type=Path, required=True, help="Path to validation pair_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints, logs, visuals")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--base-channels", type=int, default=64, help="U-Net base channels (default: 64)")
    parser.add_argument("--image-size", type=int, default=256, help="Image resize dimension (default: 256)")
    parser.add_argument("--lambda-recon", type=float, default=0.0, help="Weight for reconstruction consistency loss (default: 0.0)")
    parser.add_argument("--lambda-perc", type=float, default=0.0, help="Weight for perceptual loss (default: 0.0)")
    parser.add_argument("--seed", type=int, default=585, help="Random seed (default: 585)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples (for debugging)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples (for debugging)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "visuals").mkdir(exist_ok=True)

    logging.basicConfig(filename=args.output_dir / "train.log", level=logging.INFO, format='%(message)s')
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    train_ds = SavedSyntheticDataset(args.train_manifest, image_size=args.image_size)
    val_ds = SavedSyntheticDataset(args.val_manifest, image_size=args.image_size)

    if args.max_train_samples:
        train_ds.rows = train_ds.rows[:args.max_train_samples]
    if args.max_val_samples:
        val_ds.rows = val_ds.rows[:args.max_val_samples]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_saved)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_saved)

    model = DualHeadUNet(base_channels=args.base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    vgg_loss = None
    if args.lambda_perc > 0.0:
        vgg_loss = VGGLoss(device=device)

    log_csv = open(args.output_dir / "train_log.csv", "w", newline="")
    csv_writer = csv.writer(log_csv)
    csv_writer.writerow(["epoch", "train_loss", "train_recon_loss", "train_perc_loss", "val_loss", "lr"])

    best_val_loss = float('inf')
    
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_perc_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Train")
        
        for batch in pbar:
            mix = batch["mixture"].to(device)
            s1 = batch["source_1"].to(device)
            s2 = batch["source_2"].to(device)
            
            optimizer.zero_grad()
            p1, p2 = model(mix)
            
            pit_loss, loss_a, loss_b, is_a = permutation_invariant_l1_loss(p1, p2, s1, s2)
            
            total_loss = pit_loss
            recon_loss = torch.tensor(0.0)
            perc_loss = torch.tensor(0.0)
            
            if args.lambda_recon > 0.0:
                alphas = [float(m.get('alpha', -1.0)) for m in batch['metadata']]
                alphas_t = torch.tensor(alphas, dtype=torch.float32, device=device)
                recon_loss = get_reconstruction_loss(p1, p2, mix, alphas_t, is_a)
                total_loss += args.lambda_recon * recon_loss
                
            if args.lambda_perc > 0.0 and vgg_loss is not None:
                perc_loss = get_perceptual_loss(p1, p2, s1, s2, is_a, vgg_loss)
                total_loss += args.lambda_perc * perc_loss
                
            total_loss.backward()
            optimizer.step()
            
            train_loss += pit_loss.item() * mix.size(0)
            train_recon_loss += recon_loss.item() * mix.size(0)
            train_perc_loss += perc_loss.item() * mix.size(0)
            pbar.set_postfix({"loss": f"{pit_loss.item():.4f}", "recon": f"{recon_loss.item():.4f}", "perc": f"{perc_loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        train_recon_loss /= len(train_ds)
        train_perc_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                mix = batch["mixture"].to(device)
                s1 = batch["source_1"].to(device)
                s2 = batch["source_2"].to(device)
                
                p1, p2 = model(mix)
                pit_loss, loss_a, loss_b, is_a = permutation_invariant_l1_loss(p1, p2, s1, s2)
                
                val_loss += pit_loss.item() * mix.size(0)
                
                if i == 0:
                    p1_ordered = torch.where(is_a.view(-1, 1, 1, 1), p1, p2)
                    p2_ordered = torch.where(is_a.view(-1, 1, 1, 1), p2, p1)
                    e1 = torch.abs(s1 - p1_ordered)
                    e2 = torch.abs(s2 - p2_ordered)
                    grid = create_grid(mix.cpu(), s1.cpu(), s2.cpu(), p1_ordered.cpu(), p2_ordered.cpu(), e1.cpu(), e2.cpu())
                    vutils.save_image(grid, args.output_dir / "visuals" / f"epoch_{epoch:03d}_samples.png")
                    
        val_loss /= len(val_ds)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        logging.info(f"Epoch {epoch} | Train L1: {train_loss:.4f} | Train Recon: {train_recon_loss:.4f} | Train Perc: {train_perc_loss:.4f} | Val L1: {val_loss:.4f}")
        csv_writer.writerow([epoch, train_loss, train_recon_loss, train_perc_loss, val_loss, args.lr])
        log_csv.flush()
        plot_loss_curve(train_loss_history, val_loss_history, args.output_dir / "loss_curve.png")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args)
        }
        
        torch.save(checkpoint, args.output_dir / "checkpoints" / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, args.output_dir / "checkpoints" / "best.pt")

    log_csv.close()
    
    summary = {
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "base_channels": args.base_channels,
        "image_size": args.image_size,
        "seed": args.seed,
        "lambda_recon": args.lambda_recon,
        "lambda_perc": args.lambda_perc,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss_history[-1] if train_loss_history else None,
        "final_val_loss": val_loss_history[-1] if val_loss_history else None,
        "checkpoint_path": str(args.output_dir / "checkpoints" / "best.pt"),
        "datetime": str(datetime.datetime.now()),
        "device": str(device),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
    }

    with open(args.output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()
