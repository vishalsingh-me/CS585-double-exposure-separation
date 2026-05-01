#!/usr/bin/env python3
"""
Generate the CS585 Double-Exposure Separation starter presentation (PPTX).
Run:  python3 scripts/generate_presentation.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# ──────────────────────────────────────────────────────────
# Colour palette & helpers
# ──────────────────────────────────────────────────────────
BG_DARK  = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT   = RGBColor(0x4E, 0x9A, 0xF5)   # calm blue
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
GRAY     = RGBColor(0xAA, 0xAA, 0xBB)
ORANGE   = RGBColor(0xF5, 0xA6, 0x23)


def _set_bg(slide, color):
    """Set a solid background on a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_text_box(slide, left, top, width, height, text, *,
                  font_size=18, bold=False, italic=False,
                  color=WHITE, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.alignment = alignment
    return tf


def _add_bullet(tf, text, *, level=0, font_size=16, color=WHITE, bold=False):
    p = tf.add_paragraph()
    p.text = text
    p.level = level
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    return p


def _add_placeholder_box(slide, left, top, width, height, label):
    """Rounded-rectangle placeholder for an image insertion spot."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x2C, 0x2C, 0x44)
    shape.line.color.rgb = ACCENT
    shape.line.width = Pt(1.5)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(12)
    p.font.color.rgb = GRAY
    p.font.italic = True
    p.alignment = PP_ALIGN.CENTER


def _set_notes(slide, text):
    notes = slide.notes_slide
    notes.notes_text_frame.text = text


# ──────────────────────────────────────────────────────────
# Slide builders
# ──────────────────────────────────────────────────────────

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 1.0, 1.5, 8.0, 1.0,
                  "Separation of Double Exposure",
                  font_size=36, bold=True, color=WHITE,
                  alignment=PP_ALIGN.CENTER)

    _add_text_box(slide, 1.0, 2.7, 8.0, 0.5,
                  "CS585 · Image and Video Computing · Boston University",
                  font_size=18, color=ACCENT,
                  alignment=PP_ALIGN.CENTER)

    _add_text_box(slide, 1.0, 3.5, 8.0, 0.5,
                  "Vishal Singh  ·  Angelina Sun",
                  font_size=20, color=WHITE,
                  alignment=PP_ALIGN.CENTER)

    _add_text_box(slide, 1.0, 4.5, 8.0, 0.5,
                  "Instructor: [Name]  |  Date: [Date]",
                  font_size=14, italic=True, color=GRAY,
                  alignment=PP_ALIGN.CENTER)

    _set_notes(slide,
        "Welcome everyone. Today we are presenting our CS585 project on "
        "Separation of Double Exposure. We will cover the problem formulation, "
        "our synthetic data pipeline, baseline model, preliminary results, "
        "and next steps.")


def slide_motivation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 0.5, 0.3, 9.0, 0.7,
                  "Problem Motivation",
                  font_size=30, bold=True, color=ACCENT,
                  alignment=PP_ALIGN.LEFT)

    tf = _add_text_box(slide, 0.5, 1.2, 5.0, 3.0, "", font_size=16, color=WHITE)
    _add_bullet(tf, "Double-exposure images combine two hidden source images "
                     "into a single photograph.", font_size=16, color=WHITE)
    _add_bullet(tf, "Goal: recover the two original source images from one "
                     "blended image.", font_size=16, color=WHITE)
    _add_bullet(tf, "This is an ill-posed inverse problem — infinitely many "
                     "pairs could produce the same mixture.", font_size=16, color=WHITE)
    _add_bullet(tf, "Applications: photography restoration, forensics, "
                     "computational photography.", font_size=14, color=GRAY)

    # Diagram placeholder
    _add_text_box(slide, 0.5, 4.5, 9.0, 1.0,
        "Source 1  +  Source 2   →   Double-Exposure Mixture   →   Separation Model   →   Predicted Source 1 + 2",
        font_size=14, italic=True, color=ORANGE,
        alignment=PP_ALIGN.CENTER)

    _add_placeholder_box(slide, 6.0, 1.2, 3.5, 3.0,
        "[Insert: motivation figure\n"
        "e.g. real double-exposure example]")

    _set_notes(slide,
        "Double exposure is a photographic technique — or artifact — where "
        "two scenes are superimposed. Reversing this process is ill-posed "
        "because multiple image pairs can create the same mixture. Our project "
        "trains a neural network to invert this blending process using "
        "synthetic supervision.")


def slide_pipeline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 0.5, 0.3, 9.0, 0.7,
                  "Data Generation Pipeline",
                  font_size=30, bold=True, color=ACCENT)

    tf = _add_text_box(slide, 0.5, 1.2, 9.0, 3.5, "", font_size=16, color=WHITE)
    steps = [
        "1. Start with clean Places365 scene images (train / val split).",
        "2. Sample two source images per pair.",
        "3. Apply blending:  alpha · source₁  +  (1−α) · source₂.",
        "4. Optionally add: mask blending, Gaussian blur, gamma correction, "
         "spatial translation, additive noise.",
        "5. Save mixture image, source targets, and metadata CSV.",
    ]
    for s in steps:
        _add_bullet(tf, s, font_size=15, color=WHITE)

    _add_bullet(tf, "", font_size=8)
    _add_bullet(tf, "• Supervised data is entirely synthetic — real double "
                     "exposures lack ground-truth layers.",
                font_size=14, color=GRAY)
    _add_bullet(tf, "• Rich metadata (alpha, mode, blur, gamma) enables "
                     "interpretable evaluation.",
                font_size=14, color=GRAY)

    _set_notes(slide,
        "We cannot train with real double-exposure photos because we do not "
        "have ground-truth source layers. Instead we synthetically generate "
        "mixtures from Places365. This gives us perfect target images. "
        "We store rich metadata so we can later break down model performance "
        "by blending difficulty.")


def slide_dataset_examples(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 0.5, 0.3, 9.0, 0.7,
                  "Synthetic Dataset Examples",
                  font_size=30, bold=True, color=ACCENT)

    # Three placeholder boxes for source 1 | source 2 | mixture
    labels = ["Source 1", "Source 2", "Mixture"]
    for j, lbl in enumerate(labels):
        _add_placeholder_box(slide, 0.6 + j * 3.1, 1.3, 2.7, 2.7,
                             f"[Insert: {lbl}]")

    # Second row: another example
    for j, lbl in enumerate(labels):
        _add_placeholder_box(slide, 0.6 + j * 3.1, 4.3, 2.7, 2.1,
                             f"[Insert: {lbl}]")

    _add_text_box(slide, 0.5, 7.0, 9.0, 0.4,
        "Replace placeholders with images from results/dataset_sample_check.png",
        font_size=11, italic=True, color=GRAY,
        alignment=PP_ALIGN.CENTER)

    _set_notes(slide,
        "Here are examples of our synthetic dataset. Each row shows the two "
        "source images and the resulting mixture. The model's job is to take "
        "the mixture column and predict the source columns. Replace these "
        "placeholders with cropped samples from the generated dataset grid.")


def slide_model(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 0.5, 0.3, 9.0, 0.7,
                  "Baseline Model: Dual-Head U-Net",
                  font_size=30, bold=True, color=ACCENT)

    tf = _add_text_box(slide, 0.5, 1.2, 5.5, 2.5, "", font_size=16, color=WHITE)
    _add_bullet(tf, "Input:  1 × mixture image (3 × 256 × 256)",
                font_size=15, color=WHITE)
    _add_bullet(tf, "Shared encoder  (4 down-sampling stages)",
                font_size=15, color=WHITE)
    _add_bullet(tf, "Shared decoder  (4 up-sampling stages with skip connections)",
                font_size=15, color=WHITE)
    _add_bullet(tf, "Output:  6-channel tensor  →  split into pred_1, pred_2",
                font_size=15, color=WHITE)
    _add_bullet(tf, "Sigmoid activation  →  outputs in [0, 1]",
                font_size=15, color=WHITE)

    _add_bullet(tf, "", font_size=6)
    _add_bullet(tf, "Permutation-Invariant L1 Loss:",
                font_size=15, bold=True, color=ORANGE)
    _add_bullet(tf, "loss = min( L1(p1,s1)+L1(p2,s2) ,  L1(p1,s2)+L1(p2,s1) )",
                font_size=13, color=GRAY, bold=False)
    _add_bullet(tf, "Handles output-order ambiguity — the model is free to "
                     "assign either source to either head.",
                font_size=13, color=GRAY)

    # Architecture diagram placeholder
    _add_placeholder_box(slide, 6.3, 1.2, 3.2, 5.0,
        "[Insert: U-Net architecture diagram]\n\n"
        "Mixture → Encoder → Decoder\n"
        "                 ↘ Head 1 → Pred Source 1\n"
        "                 ↘ Head 2 → Pred Source 2")

    _set_notes(slide,
        "Our baseline is a standard U-Net with a shared encoder-decoder path. "
        "The final 1×1 conv produces 6 channels which we split into two RGB "
        "predictions. Because the source ordering is ambiguous — pred_1 could "
        "match either source — we use a permutation-invariant L1 loss that "
        "evaluates both assignments and takes the minimum.")


def slide_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 0.5, 0.3, 9.0, 0.7,
                  "Preliminary Results & Findings",
                  font_size=30, bold=True, color=ACCENT)

    # Left column: diagnostics & metrics
    tf = _add_text_box(slide, 0.5, 1.2, 5.0, 4.5, "", font_size=15, color=WHITE)
    _add_bullet(tf, "Pipeline Validation", font_size=16, bold=True, color=ORANGE)
    _add_bullet(tf, "✓  Dataset diagnostics passed — tensors RGB, float32, [0,1]",
                font_size=13, color=WHITE)
    _add_bullet(tf, "✓  End-to-end training + evaluation pipeline functional",
                font_size=13, color=WHITE)
    _add_bullet(tf, "", font_size=6)

    _add_bullet(tf, "Overfit Test  (100 epochs, 16 samples, base_ch=32)", font_size=16, bold=True, color=ORANGE)
    _add_bullet(tf, "Train L1:  0.467 → 0.174   |   Best Val L1:  0.130",
                font_size=13, color=WHITE)
    _add_bullet(tf, "Model successfully memorises small dataset — confirms pipeline works",
                font_size=13, color=WHITE)
    _add_bullet(tf, "", font_size=6)

    _add_bullet(tf, "Quick Baseline  (5 epochs, base_ch=16, full data)", font_size=16, bold=True, color=ORANGE)
    _add_bullet(tf, "Train L1:  0.430   |   Val L1:  0.421",
                font_size=13, color=WHITE)
    _add_bullet(tf, "PSNR:  12.24 dB   |   SSIM:  0.286",
                font_size=13, color=WHITE)
    _add_bullet(tf, "", font_size=6)

    _add_bullet(tf, "Stronger training planned  (32–64 channels, 30+ epochs, more data)",
                font_size=13, color=GRAY)

    # Placeholder for visuals
    _add_placeholder_box(slide, 5.8, 1.2, 3.7, 2.5,
        "[Insert: epoch_100_samples.png\nfrom experiments/overfit_test/visuals/]")
    _add_placeholder_box(slide, 5.8, 4.0, 3.7, 2.5,
        "[Insert: loss_curve.png\nfrom experiments/overfit_test/]")

    _set_notes(slide,
        "We ran a 100-epoch overfit sanity check on 16 samples with "
        "base_channels=32. The loss dropped from 0.467 to 0.174, and the "
        "best validation loss reached 0.130. The epoch 100 visual shows the "
        "model is learning to separate structure and color from the mixture. "
        "We also ran a quick 5-epoch baseline on the full dataset with "
        "base_channels=16 which achieved 12.2 dB PSNR and 0.29 SSIM. "
        "These are preliminary — stronger training with more capacity and "
        "epochs is planned.")


def slide_nextsteps(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, BG_DARK)

    _add_text_box(slide, 0.5, 0.3, 9.0, 0.7,
                  "Next Steps & Conclusion",
                  font_size=30, bold=True, color=ACCENT)

    tf = _add_text_box(slide, 0.5, 1.2, 5.5, 4.5, "", font_size=15, color=WHITE)
    _add_bullet(tf, "Planned Improvements", font_size=17, bold=True, color=ORANGE)
    _add_bullet(tf, "Train stronger baseline: base_channels = 32 → 64, 30+ epochs",
                font_size=14, color=WHITE)
    _add_bullet(tf, "Scale training data to 1,000–5,000 synthetic pairs",
                font_size=14, color=WHITE)
    _add_bullet(tf, "Add reconstruction-consistency loss  (α · p₁ + (1−α) · p₂ ≈ mixture)",
                font_size=14, color=WHITE)
    _add_bullet(tf, "Evaluate by alpha range and synthesis mode",
                font_size=14, color=WHITE)
    _add_bullet(tf, "Compare harder regimes: blur, mask, gamma, transform",
                font_size=14, color=WHITE)
    _add_bullet(tf, "", font_size=6)

    _add_bullet(tf, "Conclusion", font_size=17, bold=True, color=ORANGE)
    _add_bullet(tf,
        "This project builds a controlled synthetic framework for "
        "double-exposure separation and analyses which visual factors "
        "make separation difficult.",
        font_size=14, color=WHITE, bold=False)

    _add_placeholder_box(slide, 6.3, 1.2, 3.2, 5.0,
        "[Insert: comparison grid\nfrom stronger training run]")

    _set_notes(slide,
        "Moving forward, we will scale our training, evaluate against more "
        "challenging synthetic regimes, and study exactly what factors — "
        "alpha range, blur, gamma, spatial shift — make separation easier or "
        "harder. The key contribution of this project is the controlled "
        "synthetic framework itself, which lets us measure separation "
        "difficulty precisely. Thank you for listening.")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def create_presentation(output_path):
    prs = Presentation()
    # Widescreen 16:9
    prs.slide_width  = Emu(12192000)
    prs.slide_height = Emu(6858000)

    slide_title(prs)
    slide_motivation(prs)
    slide_pipeline(prs)
    slide_dataset_examples(prs)
    slide_model(prs)
    slide_results(prs)
    slide_nextsteps(prs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"✓ Presentation saved to {output_path}")
    print(f"  Slides: {len(prs.slides)}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "docs", "presentation",
                               "CS585_Double_Exposure_Separation_Starter.pptx")
    create_presentation(output_path)
