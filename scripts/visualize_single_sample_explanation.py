import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_factory import build_model
from src.visualization_utils import draw_centered_text, load_font, resolve_manifest_path, tensor_to_pil


def load_rgb(manifest, row, key, size):
    return Image.open(resolve_manifest_path(manifest, row[key])).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def get_predictions(args, mixture):
    if args.checkpoint is None:
        return None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_channels = int(checkpoint.get("args", {}).get("base_channels", args.base_channels))
    model_name = checkpoint.get("model") or checkpoint.get("args", {}).get("model") or args.model
    model = build_model(model_name, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        p1, p2 = model(TF.to_tensor(mixture).unsqueeze(0).to(device))
    return tensor_to_pil(p1[0], mixture.size[0]), tensor_to_pil(p2[0], mixture.size[0])


def placeholder(size, label):
    img = Image.new("RGB", (size, size), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(120, 130, 145), width=2)
    draw_centered_text(draw, (8, 8, size - 8, size - 8), label, load_font(16, bold=True), fill=(70, 75, 85))
    return img


def main():
    parser = argparse.ArgumentParser(description="Create a one-page double-exposure explanation figure.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", choices=["dual_head_unet", "two_decoder_unet"], default="dual_head_unet")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        row = list(csv.DictReader(f))[args.sample_index]

    size = 145
    s1 = load_rgb(args.manifest, row, "target_1_path", size)
    s2 = load_rgb(args.manifest, row, "target_2_path", size)
    mix = load_rgb(args.manifest, row, "mixture_path", size)
    preds = get_predictions(args, mix)
    if preds is None:
        preds = (placeholder(size, "Predicted\nSource 1"), placeholder(size, "Predicted\nSource 2"))

    width, height = 940, 890
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    step_font = load_font(20, bold=True)
    label_font = load_font(15, bold=True)
    body_font = load_font(15)
    symbol_font = load_font(30, bold=True)

    draw_centered_text(draw, (0, 20, width, 68), "Double-Exposure Separation: One-Sample Explanation", title_font)
    draw.text((50, 92), "Step 1: Original Images", font=step_font, fill=(20, 20, 20))
    canvas.paste(s1, (170, 130))
    canvas.paste(s2, (625, 130))
    draw_centered_text(draw, (170, 280, 315, 310), "Source 1", label_font)
    draw_centered_text(draw, (625, 280, 770, 310), "Source 2", label_font)

    draw.text((50, 340), "Step 2: Synthetic Forward Blend", font=step_font, fill=(20, 20, 20))
    alpha = row.get("alpha", "alpha")
    try:
        alpha_text = f"{float(alpha):.2f}"
    except (TypeError, ValueError):
        alpha_text = "alpha"
    draw_centered_text(draw, (80, 378, 860, 420), f"I = {alpha_text} * Source 1 + (1 - {alpha_text}) * Source 2", body_font)
    draw_centered_text(draw, (410, 430, 530, 470), "->", symbol_font)
    canvas.paste(mix, (397, 475))
    draw_centered_text(draw, (397, 625, 542, 650), "Blended Mixture", label_font)

    draw.text((50, 675), "Step 3: Supervised Inverse Learning", font=step_font, fill=(20, 20, 20))
    canvas.paste(mix, (88, 715))
    draw.rounded_rectangle((295, 748, 445, 805), radius=8, outline=(80, 80, 80), width=2)
    draw_centered_text(draw, (295, 748, 445, 805), "Model", label_font)
    canvas.paste(preds[0], (520, 715))
    canvas.paste(preds[1], (700, 715))
    draw_centered_text(draw, (235, 750, 285, 800), "->", symbol_font)
    draw_centered_text(draw, (455, 750, 505, 800), "->", symbol_font)

    draw.text((50, 865), "Step 4: Training targets are the original source images shown in Step 1.", font=body_font, fill=(30, 30, 30))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
