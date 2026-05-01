import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import csv
import logging
from tqdm import tqdm
import math

from src.models.dual_head_unet import DualHeadUNet
from src.losses import permutation_invariant_l1_loss
from src.data.saved_synthetic_dataset import SavedSyntheticDataset, custom_collate_saved
import torchvision.utils as vutils
from src.visualization_utils import save_separation_grid


try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Dual-Head U-Net checkpoint on a synthetic validation set.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to pair_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Image resize dimension")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = SavedSyntheticDataset(args.manifest, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_saved)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_channels = checkpoint["args"].get("base_channels", 64)
    
    model = DualHeadUNet(base_channels=base_channels).to(device)
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
            loss, loss_a, loss_b, is_a = permutation_invariant_l1_loss(p1, p2, s1, s2)
            
            p1_ordered = torch.where(is_a.view(-1, 1, 1, 1), p1, p2)
            p2_ordered = torch.where(is_a.view(-1, 1, 1, 1), p2, p1)
            
            for i in range(mix.size(0)):
                psnr_1 = compute_psnr(p1_ordered[i], s1[i])
                psnr_2 = compute_psnr(p2_ordered[i], s2[i])
                avg_psnr = (psnr_1 + psnr_2) / 2
                
                res = {
                    "l1_loss": min(loss_a[i].item(), loss_b[i].item()),
                    "psnr": avg_psnr
                }
                
                if ssim is not None:
                    p1_np = torch.clamp(p1_ordered[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    s1_np = torch.clamp(s1[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    p2_np = torch.clamp(p2_ordered[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    s2_np = torch.clamp(s2[i], 0, 1).cpu().numpy().transpose(1, 2, 0)
                    
                    ssim_1 = ssim(p1_np, s1_np, data_range=1.0, channel_axis=-1)
                    ssim_2 = ssim(p2_np, s2_np, data_range=1.0, channel_axis=-1)
                    res["ssim"] = float((ssim_1 + ssim_2) / 2)
                
                if visual_batch is None:
                    visual_batch = (
                        s1.detach().cpu(),
                        s2.detach().cpu(),
                        mix.detach().cpu(),
                        p1_ordered.detach().cpu(),
                        p2_ordered.detach().cpu(),
                    )
                
                # Safely merge metadata (guard against missing keys)
                meta = batch["metadata"][i] if i < len(batch["metadata"]) else {}
                for k, v in meta.items():
                    if k not in res:
                        res[k] = v
                results.append(res)
    
    if visual_batch is not None:
        save_separation_grid(args.output_dir / "eval_samples.png", *visual_batch)

    avg_l1 = sum([r["l1_loss"] for r in results]) / len(results)
    avg_psnr = sum([r["psnr"] for r in results]) / len(results)
    
    metrics = {
        "l1_loss": avg_l1,
        "psnr": avg_psnr,
        "num_samples": len(results)
    }
    
    if ssim is not None:
        metrics["ssim"] = sum([r["ssim"] for r in results]) / len(results)
        
    with open(args.output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    all_keys = list(results[0].keys()) if results else []
    with open(args.output_dir / "eval_per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)

    # Also write the old filename for backward compatibility
    with open(args.output_dir / "eval_by_mode.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)
        
    # Grouped Metrics Helper
    def group_by_key(key, transform_fn=lambda x: x):
        grouped = {}
        for r in results:
            val = r.get(key)
            if val is not None and val != "":
                val = transform_fn(val)
                if val not in grouped:
                    grouped[val] = {"count": 0, "l1_loss": 0, "psnr": 0, "ssim": 0}
                grouped[val]["count"] += 1
                grouped[val]["l1_loss"] += r["l1_loss"]
                grouped[val]["psnr"] += r["psnr"]
                if "ssim" in r:
                    grouped[val]["ssim"] += r["ssim"]
                    
        summary_rows = []
        for k, v in grouped.items():
            row = {key: k, "count": v["count"], "l1_loss": v["l1_loss"] / v["count"], "psnr": v["psnr"] / v["count"]}
            if ssim is not None:
                row["ssim"] = v["ssim"] / v["count"]
            summary_rows.append(row)
        return summary_rows

    def get_alpha_bin(alpha):
        try:
            a = float(alpha)
            if 0.3 <= a < 0.4: return "0.30-0.40"
            elif 0.4 <= a < 0.5: return "0.40-0.50"
            elif 0.5 <= a < 0.6: return "0.50-0.60"
            elif 0.6 <= a <= 0.7: return "0.60-0.70"
            return "Other"
        except:
            return "Unknown"

    alpha_rows = group_by_key("alpha", get_alpha_bin)
    if alpha_rows:
        with open(args.output_dir / "eval_by_alpha_bin.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=alpha_rows[0].keys())
            writer.writeheader()
            writer.writerows(alpha_rows)

    mode_rows = group_by_key("synthesis_mode")
    if mode_rows:
        with open(args.output_dir / "eval_by_synthesis_mode.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mode_rows[0].keys())
            writer.writeheader()
            writer.writerows(mode_rows)
                
    label_rows = group_by_key("same_label")
    if label_rows:
        with open(args.output_dir / "eval_by_same_label.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=label_rows[0].keys())
            writer.writeheader()
            writer.writerows(label_rows)
        
    print(f"\nEvaluation complete.")
    print(f"  Samples: {len(results)}")
    print(f"  L1: {avg_l1:.4f}  |  PSNR: {avg_psnr:.2f} dB", end="")
    if ssim is not None and 'ssim' in metrics:
        print(f"  |  SSIM: {metrics['ssim']:.4f}")
    else:
        print()
    print(f"  Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
