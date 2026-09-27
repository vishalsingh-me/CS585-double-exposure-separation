import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from pathlib import Path
import csv
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser(
        description="Check dataset samples: print tensor stats and save a diagnostic grid.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to pair_manifest.csv")
    parser.add_argument("--output", type=Path, default=Path("results/dataset_sample_check.png"), help="Output image path")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to visualise")
    args = parser.parse_args()
    
    with open(args.manifest, 'r') as f:
        rows = list(csv.DictReader(f))
        
    num_samples = min(args.num_samples, len(rows))
    rows = rows[:num_samples]
    
    base_dir = args.manifest.parent
    
    images_to_concat = []
    
    print("-" * 50)
    for i, row in enumerate(rows):
        p_mix = base_dir / row["mixture_path"]
        p_s1 = base_dir / row["target_1_path"]
        p_s2 = base_dir / row["target_2_path"]
        
        mix_img = Image.open(p_mix).convert('RGB')
        s1_img = Image.open(p_s1).convert('RGB')
        s2_img = Image.open(p_s2).convert('RGB')
        
        t_mix = TF.to_tensor(mix_img)
        t_s1 = TF.to_tensor(s1_img)
        t_s2 = TF.to_tensor(s2_img)
        
        print(f"Sample {i+1}:")
        for name, t, p in [("Mixture", t_mix, p_mix), ("Source 1", t_s1, p_s1), ("Source 2", t_s2, p_s2)]:
            print(f"  {name}: {p.exists()} {p}")
            print(f"    Shape: {t.shape}, Dtype: {t.dtype}")
            print(f"    Min: {t.min().item():.3f}, Max: {t.max().item():.3f}, Mean: {t.mean().item():.3f}")
        
        print(f"  Metadata: alpha={row.get('alpha', 'N/A')}, mode={row.get('synthesis_mode', 'N/A')}")
        print("-" * 50)
        
        images_to_concat.extend([
            torch.clamp(t_s1, 0, 1),
            torch.clamp(t_s2, 0, 1),
            torch.clamp(t_mix, 0, 1),
        ])
        
    if images_to_concat:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        grid = vutils.make_grid(images_to_concat, nrow=3, normalize=False, padding=2)
        vutils.save_image(grid, args.output)
        print(f"\nSaved diagnostic grid ({num_samples} samples) to {args.output}")
    print("Dataset sample check complete.")

if __name__ == "__main__":
    main()