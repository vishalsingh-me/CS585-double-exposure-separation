import os
import argparse
import csv
import random
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_preview.png"))
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()
    
    with open(args.manifest, 'r') as f:
        rows = list(csv.DictReader(f))
        
    if len(rows) > args.num_samples:
        rows = random.sample(rows, args.num_samples)
        
    base_dir = args.manifest.parent
    
    images_to_concat = []
    
    for row in rows:
        m = Image.open(base_dir / row["mixture_path"])
        s1 = Image.open(base_dir / row["target_1_path"])
        s2 = Image.open(base_dir / row["target_2_path"])
        
        row_imgs = [s1, s2, m]
        if row.get("mask_path"):
            mask_path = base_dir / row["mask_path"]
            if mask_path.exists():
                row_imgs.append(Image.open(mask_path).convert("RGB"))
                
        # concat row
        widths, heights = zip(*(i.size for i in row_imgs))
        total_width = sum(widths)
        max_height = max(heights)
        
        row_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in row_imgs:
            row_img.paste(im, (x_offset, 0))
            x_offset += im.size[0]
            
        images_to_concat.append(row_img)
        
    if images_to_concat:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        total_width = max(i.size[0] for i in images_to_concat)
        total_height = sum(i.size[1] for i in images_to_concat)
        
        grid = Image.new('RGB', (total_width, total_height))
        y_offset = 0
        for im in images_to_concat:
            grid.paste(im, (0, y_offset))
            y_offset += im.size[1]
            
        grid.save(args.output)
        print(f"Saved preview to {args.output}")

if __name__ == "__main__":
    main()
