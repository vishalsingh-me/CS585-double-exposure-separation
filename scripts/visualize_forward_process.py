import argparse
import csv
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualization_utils import draw_centered_text, load_font, resolve_manifest_path


def load_image(manifest, row, key, size):
    return Image.open(resolve_manifest_path(manifest, row[key])).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic double-exposure generation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))
    rows = rows[: args.num_samples] if len(rows) <= args.num_samples else random.sample(rows, args.num_samples)

    image_size = 180
    margin = 34
    gap = 28
    symbol_w = 42
    title_h = 62
    label_h = 34
    meta_h = 46
    row_h = label_h + image_size + meta_h
    width = margin * 2 + image_size * 3 + symbol_w * 2 + gap * 4
    height = margin + title_h + len(rows) * row_h + margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    label_font = load_font(16, bold=True)
    meta_font = load_font(14)
    symbol_font = load_font(34, bold=True)

    draw_centered_text(draw, (0, margin, width, margin + title_h), "Synthetic Double-Exposure Generation", title_font)
    y = margin + title_h
    for row in rows:
        s1 = load_image(args.manifest, row, "target_1_path", image_size)
        s2 = load_image(args.manifest, row, "target_2_path", image_size)
        mix = load_image(args.manifest, row, "mixture_path", image_size)

        x1 = margin
        plus_x = x1 + image_size + gap
        x2 = plus_x + symbol_w + gap
        arrow_x = x2 + image_size + gap
        xm = arrow_x + symbol_w + gap

        draw_centered_text(draw, (x1, y, x1 + image_size, y + label_h), "Original Source 1", label_font)
        draw_centered_text(draw, (x2, y, x2 + image_size, y + label_h), "Original Source 2", label_font)
        draw_centered_text(draw, (xm, y, xm + image_size, y + label_h), "Blended Mixture", label_font)
        canvas.paste(s1, (x1, y + label_h))
        canvas.paste(s2, (x2, y + label_h))
        canvas.paste(mix, (xm, y + label_h))
        draw_centered_text(draw, (plus_x, y + label_h, plus_x + symbol_w, y + label_h + image_size), "+", symbol_font)
        draw_centered_text(draw, (arrow_x, y + label_h, arrow_x + symbol_w, y + label_h + image_size), "->", symbol_font)
        alpha = row.get("alpha", "")
        mode = row.get("synthesis_mode", "")
        try:
            alpha = f"{float(alpha):.2f}"
        except (TypeError, ValueError):
            alpha = "unknown"
        draw_centered_text(draw, (xm, y + label_h + image_size, xm + image_size, y + row_h), f"alpha = {alpha}\nmode = {mode}", meta_font)
        y += row_h

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
