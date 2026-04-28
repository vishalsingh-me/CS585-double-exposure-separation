import os
import argparse
import random
import csv
import logging
from pathlib import Path
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

def configure_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def load_rgb_image(path):
    return Image.open(path).convert('RGB')

def resize_center_crop(image, size):
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = image.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((size, size), Image.Resampling.LANCZOS)

def sample_pair(manifest, rng, same_label_probability, current_split):
    # filter manifest
    valid_rows = [row for row in manifest if row.get("split", current_split) == current_split]
    if len(valid_rows) < 2:
        return None, None, False
    
    label_dict = {}
    for row in valid_rows:
        lbl = row.get("label", "unknown")
        if lbl not in label_dict:
            label_dict[lbl] = []
        label_dict[lbl].append(row)
        
    same_label = False
    img1 = rng.choice(valid_rows)
    img2 = None
    
    if rng.random() < same_label_probability and len(label_dict.get(img1.get("label", "unknown"), [])) > 1:
        same_label = True
        candidates = [row for row in label_dict[img1.get("label", "unknown")] if row != img1]
        if candidates:
            img2 = rng.choice(candidates)
            
    if img2 is None:
        same_label = False
        candidates = [row for row in valid_rows if row != img1]
        img2 = rng.choice(candidates)
        
    return img1, img2, same_label

def apply_translation(image, dx, dy):
    return TF.affine(image, angle=0.0, translate=(dx, dy), scale=1.0, shear=0.0, interpolation=TF.InterpolationMode.BILINEAR)

def apply_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def generate_mask(height, width, mask_type, rng):
    mask = torch.zeros(1, height, width)
    if mask_type == "gradient":
        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - width/2)**2 + (y - height/2)**2)
                val = max(0, 1 - dist / (width/2))
                mask[0, y, x] = val
    else:
        # smooth random
        mask = torch.rand(1, height//16, width//16)
        mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode='bilinear').squeeze(0)
    return mask

def blend_images(img1, img2, config):
    t1 = TF.to_tensor(img1)
    t2 = TF.to_tensor(img2)
    
    # transform image 2
    if config["transform_dx"] != 0 or config["transform_dy"] != 0:
        t2 = TF.affine(t2, angle=0.0, translate=[config["transform_dx"], config["transform_dy"]], scale=1.0, shear=[0.0, 0.0])
    
    if config["blur_radius"] > 0:
        img2_blur = img2.filter(ImageFilter.GaussianBlur(config["blur_radius"]))
        t2 = TF.to_tensor(img2_blur)
        
    mask = None
    mask_stats = {}
    if config["mode"].startswith("mask"):
        mask = generate_mask(t1.shape[1], t1.shape[2], "gradient", random.Random(config["seed"]))
        mixture = mask * t1 + (1 - mask) * t2
        mask_stats["mask_mean"] = mask.mean().item()
        mask_stats["mask_min"] = mask.min().item()
        mask_stats["mask_max"] = mask.max().item()
    else:
        alpha = config["alpha"]
        mixture = alpha * t1 + (1 - alpha) * t2
        
    if config["gamma"] != 1.0:
        mixture = mixture ** config["gamma"]
        
    if config["noise_std"] > 0:
        noise = torch.randn_like(mixture) * config["noise_std"]
        mixture = torch.clamp(mixture + noise, 0, 1)
        
    return mixture, t1, t2, mask, mask_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mode", type=str, default="linear_clean")
    parser.add_argument("--alpha-min", type=float, default=0.3)
    parser.add_argument("--alpha-max", type=float, default=0.7)
    parser.add_argument("--gamma-min", type=float, default=1.0)
    parser.add_argument("--gamma-max", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--blur-min", type=float, default=0.0)
    parser.add_argument("--blur-max", type=float, default=0.0)
    parser.add_argument("--max-shift", type=int, default=0)
    parser.add_argument("--same-label-probability", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    
    with open(args.input_csv, 'r') as f:
        manifest = list(csv.DictReader(f))
        
    out_dir = args.output_dir / args.split
    (out_dir / "mixtures").mkdir(parents=True, exist_ok=True)
    (out_dir / "source_1").mkdir(parents=True, exist_ok=True)
    (out_dir / "source_2").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        
    manifest_rows = []
    
    for i in range(args.num_pairs):
        img1_meta, img2_meta, same_label = sample_pair(manifest, rng, args.same_label_probability, args.split)
        if img1_meta is None:
            continue
        
        rel1 = img1_meta.get("relative_path", img1_meta.get("path", ""))
        rel2 = img2_meta.get("relative_path", img2_meta.get("path", ""))
        
        path1 = args.image_root / rel1
        path2 = args.image_root / rel2
        
        i1 = resize_center_crop(load_rgb_image(path1), args.image_size)
        i2 = resize_center_crop(load_rgb_image(path2), args.image_size)
        
        config = {
            "mode": args.mode,
            "alpha": rng.uniform(args.alpha_min, args.alpha_max),
            "gamma": rng.uniform(args.gamma_min, args.gamma_max),
            "noise_std": args.noise_std,
            "blur_radius": rng.uniform(args.blur_min, args.blur_max),
            "transform_dx": rng.randint(-args.max_shift, args.max_shift) if args.max_shift > 0 else 0,
            "transform_dy": rng.randint(-args.max_shift, args.max_shift) if args.max_shift > 0 else 0,
            "seed": rng.randint(0, 1000000)
        }
        
        mix, t1, t2, mask, mask_stats = blend_images(i1, i2, config)
        
        prefix = f"sample_{i:06d}"
        mix_img = TF.to_pil_image(mix)
        mix_img.save(out_dir / "mixtures" / f"{prefix}_mix.png")
        
        s1_img = TF.to_pil_image(t1)
        s1_img.save(out_dir / "source_1" / f"{prefix}_source_1.png")
        
        s2_img = TF.to_pil_image(t2)
        s2_img.save(out_dir / "source_2" / f"{prefix}_source_2.png")
        
        mask_path = ""
        if mask is not None and args.save_masks:
            mask_img = TF.to_pil_image(mask)
            mask_img.save(out_dir / "masks" / f"{prefix}_mask.png")
            mask_path = f"masks/{prefix}_mask.png"
            
        row = {
            "pair_id": i,
            "split": args.split,
            "synthesis_mode": args.mode,
            "source_1_path": str(path1),
            "source_2_path": str(path2),
            "source_1_label": img1_meta.get("label", ""),
            "source_2_label": img2_meta.get("label", ""),
            "same_label": same_label,
            "mixture_path": f"mixtures/{prefix}_mix.png",
            "target_1_path": f"source_1/{prefix}_source_1.png",
            "target_2_path": f"source_2/{prefix}_source_2.png",
            "mask_path": mask_path,
            "image_size": args.image_size,
            "alpha": config["alpha"],
            "gamma": config["gamma"],
            "noise_std": config["noise_std"],
            "blur_radius": config["blur_radius"],
            "transform_dx": config["transform_dx"],
            "transform_dy": config["transform_dy"],
            "mask_type": "gradient" if mask is not None else "",
            "seed": config["seed"]
        }
        row.update(mask_stats)
        manifest_rows.append(row)
        
    keys = manifest_rows[0].keys() if manifest_rows else []
    with open(out_dir / "pair_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest_rows)

if __name__ == "__main__":
    main()
