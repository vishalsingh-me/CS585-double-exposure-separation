from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


def resolve_manifest_path(manifest_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(manifest_path).parent / path


def tensor_to_pil(tensor, size=None):
    image = TF.to_pil_image(torch.clamp(tensor.detach().cpu(), 0, 1))
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_text(draw, box, text, font, fill=(20, 20, 20)):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + ((right - left) - width) // 2
    y = top + ((bottom - top) - height) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4, align="center")


def match_predictions(pred_1, pred_2, source_1, source_2, is_a):
    mask = is_a.view(-1, 1, 1, 1)
    pred_1_matched = torch.where(mask, pred_1, pred_2)
    pred_2_matched = torch.where(mask, pred_2, pred_1)
    return pred_1_matched, pred_2_matched


def save_separation_grid(
    output_path,
    source_1,
    source_2,
    mixture,
    pred_1,
    pred_2,
    max_rows=4,
    title="Double-Exposure Separation: Ground Truth, Mixture, Prediction, and Error",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_1 = source_1.detach().cpu()
    source_2 = source_2.detach().cpu()
    mixture = mixture.detach().cpu()
    pred_1 = pred_1.detach().cpu()
    pred_2 = pred_2.detach().cpu()

    count = min(max_rows, source_1.size(0))
    image_size = 128
    gap = 16
    title_h = 54
    section_h = 30
    label_h = 42
    row_gap = 18
    margin = 28
    columns = [
        ("Original Source 1", "Ground Truth Sources"),
        ("Original Source 2", "Ground Truth Sources"),
        ("Blended Mixture", "Forward Process"),
        ("Predicted Source 1", "Model Output"),
        ("Predicted Source 2", "Model Output"),
        ("Error Map 1", "Error"),
        ("Error Map 2", "Error"),
    ]
    width = margin * 2 + len(columns) * image_size + (len(columns) - 1) * gap
    height = margin + title_h + section_h + label_h + count * image_size + (count - 1) * row_gap + margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    section_font = load_font(15, bold=True)
    label_font = load_font(14)

    draw_centered_text(draw, (0, margin, width, margin + title_h), title, title_font)

    x_positions = [margin + i * (image_size + gap) for i in range(len(columns))]
    section_y = margin + title_h
    section_groups = [
        ("Ground Truth Sources", 0, 1),
        ("Forward Process", 2, 2),
        ("Model Output", 3, 4),
        ("Error", 5, 6),
    ]
    for label, start, end in section_groups:
        left = x_positions[start]
        right = x_positions[end] + image_size
        draw_centered_text(draw, (left, section_y, right, section_y + section_h), label, section_font, fill=(70, 70, 70))

    label_y = section_y + section_h
    for idx, (label, _) in enumerate(columns):
        draw_centered_text(draw, (x_positions[idx], label_y, x_positions[idx] + image_size, label_y + label_h), label, label_font)

    y = label_y + label_h
    for row_idx in range(count):
        e1 = torch.clamp(torch.abs(source_1[row_idx] - pred_1[row_idx]) * 3.0, 0, 1)
        e2 = torch.clamp(torch.abs(source_2[row_idx] - pred_2[row_idx]) * 3.0, 0, 1)
        row_images = [
            source_1[row_idx],
            source_2[row_idx],
            mixture[row_idx],
            pred_1[row_idx],
            pred_2[row_idx],
            e1,
            e2,
        ]
        for col_idx, tensor in enumerate(row_images):
            canvas.paste(tensor_to_pil(tensor, image_size), (x_positions[col_idx], y))
        y += image_size + row_gap

    canvas.save(output_path)
