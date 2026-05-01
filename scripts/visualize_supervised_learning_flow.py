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


def predict(args, mixture):
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
        x = TF.to_tensor(mixture).unsqueeze(0).to(device)
        p1, p2 = model(x)
    return tensor_to_pil(p1[0], mixture.size[0]), tensor_to_pil(p2[0], mixture.size[0])


def main():
    parser = argparse.ArgumentParser(description="Visualize supervised inverse learning for double exposure.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", choices=["dual_head_unet", "two_decoder_unet"], default="dual_head_unet")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))
    row = rows[args.sample_index]

    image_size = 180
    mix = load_rgb(args.manifest, row, "mixture_path", image_size)
    s1 = load_rgb(args.manifest, row, "target_1_path", image_size)
    s2 = load_rgb(args.manifest, row, "target_2_path", image_size)
    preds = predict(args, mix)

    width, height = 860, 560 if preds else 430
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    label_font = load_font(16, bold=True)
    body_font = load_font(15)
    symbol_font = load_font(34, bold=True)

    draw_centered_text(draw, (0, 22, width, 70), "Supervised Learning Setup", title_font)
    draw_centered_text(draw, (70, 92, 250, 130), "Input X: Blended Mixture", label_font)
    canvas.paste(mix, (70, 130))
    draw_centered_text(draw, (315, 160, 545, 230), "Model:\nDual-Head U-Net", label_font)
    draw.rounded_rectangle((315, 240, 545, 300), radius=8, outline=(80, 80, 80), width=2)
    draw_centered_text(draw, (315, 240, 545, 300), "mixture -> source_1 + source_2", body_font)
    draw_centered_text(draw, (260, 170, 305, 240), "->", symbol_font)

    if preds:
        draw_centered_text(draw, (610, 92, 790, 130), "Predictions", label_font)
        canvas.paste(preds[0], (560, 130))
        canvas.paste(preds[1], (750 - image_size // 2, 130))
        draw_centered_text(draw, (560, 318, 740, 350), "Predicted Source 1", body_font)
        draw_centered_text(draw, (660, 318, 840, 350), "Predicted Source 2", body_font)
        draw_centered_text(draw, (545, 170, 590, 240), "->", symbol_font)

    y_targets = 330 if preds else 300
    draw_centered_text(draw, (0, y_targets, width, y_targets + 36), "Training Targets: original source images are ground truth labels, not model inputs", label_font)
    canvas.paste(s1, (235, y_targets + 50))
    canvas.paste(s2, (455, y_targets + 50))
    draw_centered_text(draw, (235, y_targets + 235, 415, y_targets + 265), "Target Y1: Original Source 1", body_font)
    draw_centered_text(draw, (455, y_targets + 235, 635, y_targets + 265), "Target Y2: Original Source 2", body_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
