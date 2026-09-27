#!/usr/bin/env python3
"""
Generate the FINAL CS585 Double-Exposure Separation presentation (PPTX).
Embeds actual experiment images where available.
Run:  python3 scripts/generate_final_presentation.py
"""
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = PROJECT_ROOT / "docs" / "presentation" / "assets"

# ── Palette ──
BG       = RGBColor(0x0F, 0x11, 0x1A)
BG_CARD  = RGBColor(0x1A, 0x1D, 0x2B)
ACCENT   = RGBColor(0x5B, 0xA8, 0xF5)
ACCENT2  = RGBColor(0x34, 0xD3, 0x99)
WHITE    = RGBColor(0xF0, 0xF0, 0xF5)
LGRAY    = RGBColor(0x99, 0x9D, 0xB3)
ORANGE   = RGBColor(0xF0, 0xA5, 0x30)
DCARD    = RGBColor(0x22, 0x25, 0x38)

def _bg(slide):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = BG

def _tb(slide, l, t, w, h, txt, sz=18, bold=False, italic=False, color=WHITE, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = txt
    p.font.size = Pt(sz); p.font.bold = bold; p.font.italic = italic
    p.font.color.rgb = color; p.alignment = align
    return tf

def _bp(tf, txt, sz=15, color=WHITE, bold=False, level=0):
    p = tf.add_paragraph(); p.text = txt; p.level = level
    p.font.size = Pt(sz); p.font.color.rgb = color; p.font.bold = bold
    return p

def _card(slide, l, t, w, h, color=DCARD, border=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color
    if border:
        s.line.color.rgb = border; s.line.width = Pt(1.2)
    else:
        s.line.fill.background()
    return s

def _placeholder(slide, l, t, w, h, label):
    s = _card(slide, l, t, w, h, color=RGBColor(0x1E,0x20,0x30), border=ACCENT)
    tf = s.text_frame; tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].text = label
    tf.paragraphs[0].font.size = Pt(11)
    tf.paragraphs[0].font.color.rgb = LGRAY
    tf.paragraphs[0].font.italic = True
    s.text_frame.paragraphs[0].space_before = Pt(0)

def _arrow_box(slide, l, t, w, h, txt, fill=ACCENT):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background()
    tf = s.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = txt; p.font.size = Pt(11); p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF,0xFF,0xFF); p.alignment = PP_ALIGN.CENTER
    return s

def _arrow_label(slide, l, t, txt):
    _tb(slide, l, t, 0.4, 0.3, txt, sz=16, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

def _notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt

def _try_image(slide, path, l, t, w, h, fallback_label):
    fp = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    if fp.exists():
        slide.shapes.add_picture(str(fp), Inches(l), Inches(t), Inches(w), Inches(h))
        return True
    _placeholder(slide, l, t, w, h, fallback_label)
    return False

# ─────────────────────────────────────────────
# SLIDE 1: Title
# ─────────────────────────────────────────────
def s1_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    # Accent line
    _card(slide, 3.0, 1.0, 4.0, 0.06, color=ACCENT)
    _tb(slide, 0.5, 1.3, 9.0, 1.0,
        "Separation of Double Exposure",
        sz=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _tb(slide, 0.5, 2.5, 9.0, 0.6,
        "Recovering Two Source Images from a Single Blended Image",
        sz=18, color=LGRAY, align=PP_ALIGN.CENTER)
    _card(slide, 3.0, 3.3, 4.0, 0.04, color=ACCENT)
    _tb(slide, 0.5, 3.6, 9.0, 0.5,
        "Vishal Singh  ·  Angelina Sun",
        sz=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _tb(slide, 0.5, 4.5, 9.0, 0.5,
        "CS585 Image and Video Computing  ·  Boston University",
        sz=14, color=ACCENT, align=PP_ALIGN.CENTER)
    _notes(slide,
        "Welcome. Today we present our CS585 project: Separation of Double Exposure. "
        "We built an end-to-end framework to recover two source images from a single "
        "blended double-exposure photograph. We will cover the problem, our synthetic "
        "data pipeline, model, results, and next steps.")

# ─────────────────────────────────────────────
# SLIDE 2: Problem & Goal
# ─────────────────────────────────────────────
def s2_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    _tb(slide, 0.5, 0.25, 9.0, 0.6, "Problem & Goal", sz=30, bold=True, color=ACCENT)

    tf = _tb(slide, 0.5, 1.0, 5.5, 2.0, "", sz=15, color=WHITE)
    _bp(tf, "Double-exposure images combine two scenes into one photograph.", sz=15, color=WHITE)
    _bp(tf, "Goal: recover both original source images from a single mixture.", sz=15, color=WHITE)
    _bp(tf, "This is an ill-posed inverse problem — many image pairs could produce the same blend.", sz=15, color=WHITE)
    _bp(tf, "Real double exposures lack ground-truth layers, so we use synthetic supervision.", sz=14, color=LGRAY)

    # Flow diagram row
    y = 3.5; bh = 0.6; bw = 1.6
    _arrow_box(slide, 0.3, y, bw, bh, "Source 1", fill=ACCENT2)
    _arrow_label(slide, 2.0, y+0.1, "+")
    _arrow_box(slide, 2.4, y, bw, bh, "Source 2", fill=ACCENT2)
    _arrow_label(slide, 4.1, y+0.1, "→")
    _arrow_box(slide, 4.5, y, bw, bh, "Mixture", fill=ORANGE)
    _arrow_label(slide, 6.2, y+0.1, "→")
    _arrow_box(slide, 6.6, y, bw, bh, "Model", fill=ACCENT)
    _arrow_label(slide, 8.3, y+0.1, "→")
    y2 = 4.4
    _arrow_box(slide, 6.6, y2, 1.6, 0.5, "Pred Src 1", fill=ACCENT2)
    _arrow_box(slide, 8.4, y2, 1.2, 0.5, "Pred Src 2", fill=ACCENT2)

    _placeholder(slide, 6.5, 0.9, 3.2, 2.2, "[Insert: double-exposure example image]")

    _notes(slide,
        "Double exposure is when two scenes are superimposed into one image. "
        "Reversing this is ill-posed because infinitely many pairs can create the same "
        "mixture. Since real double-exposure photos don't come with ground-truth layers, "
        "we generate synthetic training data with known source images.")

# ─────────────────────────────────────────────
# SLIDE 3: Synthetic Data Pipeline
# ─────────────────────────────────────────────
def s3_pipeline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    _tb(slide, 0.5, 0.25, 9.0, 0.6, "Synthetic Data Pipeline", sz=30, bold=True, color=ACCENT)

    # Pipeline flow boxes
    steps = [
        ("Places365\nClean Images", ACCENT2),
        ("Sample\nImage Pairs", ACCENT),
        ("Apply\nBlending", ORANGE),
        ("Save Mixture\n+ Targets\n+ Metadata", ACCENT),
    ]
    y = 1.2; bw = 2.0; bh = 0.9
    for i, (txt, col) in enumerate(steps):
        x = 0.4 + i * 2.45
        _arrow_box(slide, x, y, bw, bh, txt, fill=col)
        if i < len(steps) - 1:
            _arrow_label(slide, x + bw + 0.02, y + 0.25, "→")

    # Blending modes card
    _card(slide, 0.4, 2.5, 4.5, 2.7)
    tf = _tb(slide, 0.6, 2.6, 4.2, 2.5, "Supported Blending Modes", sz=16, bold=True, color=ORANGE)
    modes = [
        "α · source₁ + (1−α) · source₂   (linear alpha)",
        "Gradient mask blending",
        "Gaussian blur on one source",
        "Gamma correction on mixture",
        "Spatial translation / shift",
        "Additive Gaussian noise",
    ]
    for m in modes:
        _bp(tf, "•  " + m, sz=12, color=WHITE)

    # Key points card
    _card(slide, 5.2, 2.5, 4.5, 2.7)
    tf = _tb(slide, 5.4, 2.6, 4.2, 2.5, "Key Design Decisions", sz=16, bold=True, color=ACCENT2)
    _bp(tf, "Supervised data is entirely synthetic — real double exposures lack ground-truth.", sz=12, color=WHITE)
    _bp(tf, "Rich metadata (α, mode, blur, γ) enables interpretable, fine-grained evaluation.", sz=12, color=WHITE)
    _bp(tf, "Train / val splits ensure no image leakage.", sz=12, color=WHITE)
    _bp(tf, "Each sample saves: mixture.png, source_1.png, source_2.png, manifest row.", sz=12, color=WHITE)

    _notes(slide,
        "We cannot train with real double-exposure photos because we have no ground-truth "
        "source layers. Instead we synthetically generate mixtures from Places365 images. "
        "The pipeline samples two images, applies configurable blending, and saves the "
        "mixture, both source targets, and rich metadata. This metadata lets us later "
        "evaluate model performance by alpha range, blending mode, and scene similarity.")

# ─────────────────────────────────────────────
# SLIDE 4: Dataset Examples
# ─────────────────────────────────────────────
def s4_examples(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    _tb(slide, 0.5, 0.25, 9.0, 0.6, "Dataset Examples", sz=30, bold=True, color=ACCENT)
    _tb(slide, 0.5, 0.75, 9.0, 0.4,
        "Each training sample: Source 1 | Source 2 | Mixture",
        sz=14, color=LGRAY)

    found = _try_image(slide,
        "docs/presentation/assets/dataset_sample_check.png",
        0.4, 1.3, 9.2, 3.5,
        "[Insert actual result here]\nresults/dataset_sample_check.png")

    if not found:
        labels = ["Source 1", "Source 2", "Mixture"]
        for j, lbl in enumerate(labels):
            _placeholder(slide, 0.5 + j * 3.2, 1.5, 2.8, 2.5, f"[Insert: {lbl}]")

    _card(slide, 0.4, 5.1, 9.2, 1.4)
    tf = _tb(slide, 0.6, 5.2, 8.8, 1.2, "", sz=13, color=WHITE)
    _bp(tf, "The model receives the mixture as input and must predict both sources.", sz=13, color=WHITE)
    _bp(tf, "Source ordering is ambiguous — permutation-invariant loss handles this.", sz=13, color=WHITE)
    _bp(tf, "Metadata records alpha, blending mode, blur, and gamma for each pair.", sz=13, color=LGRAY)

    _notes(slide,
        "This grid shows examples from our synthetic dataset. Each row has two source "
        "images and the resulting blended mixture. The model's task is to take the mixture "
        "and predict the two source images. Note that the ordering is ambiguous — the model "
        "could output either source as pred_1 or pred_2, which is why we use permutation-invariant loss.")

# ─────────────────────────────────────────────
# SLIDE 5: Model Architecture
# ─────────────────────────────────────────────
def s5_model(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    _tb(slide, 0.5, 0.25, 9.0, 0.6, "Baseline Model: Dual-Head U-Net", sz=30, bold=True, color=ACCENT)

    # Architecture diagram with shape boxes
    # Encoder
    _arrow_box(slide, 0.3, 1.3, 1.6, 0.7, "Mixture\n3 × 256 × 256", fill=ORANGE)
    _arrow_label(slide, 2.0, 1.4, "→")
    _arrow_box(slide, 2.4, 1.1, 1.8, 1.1, "Shared\nEncoder\n4 ↓ stages", fill=ACCENT)
    _arrow_label(slide, 4.3, 1.4, "→")
    _arrow_box(slide, 4.7, 1.1, 1.8, 1.1, "Shared\nDecoder\n4 ↑ stages", fill=ACCENT)
    _arrow_label(slide, 6.6, 1.2, "→")
    # Output split
    _arrow_box(slide, 7.0, 1.0, 1.4, 0.5, "6-ch output", fill=RGBColor(0x55,0x55,0x77))
    _tb(slide, 7.0, 1.55, 0.7, 0.3, "split", sz=10, color=LGRAY, align=PP_ALIGN.CENTER)
    _arrow_box(slide, 6.7, 1.9, 1.0, 0.5, "Pred Src 1", fill=ACCENT2)
    _arrow_box(slide, 7.9, 1.9, 1.0, 0.5, "Pred Src 2", fill=ACCENT2)
    # Skip connection note
    _tb(slide, 2.7, 2.4, 3.5, 0.3, "← skip connections →", sz=10, italic=True, color=LGRAY, align=PP_ALIGN.CENTER)

    # Details cards
    _card(slide, 0.3, 3.0, 4.5, 3.2)
    tf = _tb(slide, 0.5, 3.1, 4.2, 3.0, "Architecture Details", sz=16, bold=True, color=ACCENT)
    _bp(tf, "Input: 1 mixture image  (3 × 256 × 256)", sz=13, color=WHITE)
    _bp(tf, "Encoder: C → 2C → 4C → 8C → 16C  (max-pool)", sz=13, color=WHITE)
    _bp(tf, "Decoder: transposed conv + skip connections", sz=13, color=WHITE)
    _bp(tf, "Final 1×1 conv → 6 channels → split to 2×RGB", sz=13, color=WHITE)
    _bp(tf, "Sigmoid activation → outputs in [0, 1]", sz=13, color=WHITE)
    _bp(tf, "Configurable base channels (16, 32, 64)", sz=13, color=LGRAY)

    _card(slide, 5.1, 3.0, 4.6, 3.2)
    tf = _tb(slide, 5.3, 3.1, 4.3, 3.0, "Permutation-Invariant L1 Loss", sz=16, bold=True, color=ORANGE)
    _bp(tf, "loss_a = L1(p₁, s₁) + L1(p₂, s₂)", sz=13, color=WHITE)
    _bp(tf, "loss_b = L1(p₁, s₂) + L1(p₂, s₁)", sz=13, color=WHITE)
    _bp(tf, "final_loss = min(loss_a, loss_b)", sz=14, color=ACCENT2, bold=True)
    _bp(tf, "", sz=6)
    _bp(tf, "Handles output-order ambiguity.", sz=13, color=WHITE)
    _bp(tf, "The model is free to assign either source to either output head.", sz=13, color=LGRAY)

    _notes(slide,
        "Our baseline is a Dual-Head U-Net. A shared encoder-decoder processes the mixture "
        "image. The final 1×1 conv produces 6 channels which are split into two 3-channel "
        "RGB predictions. Because the two sources are unordered, we use a permutation-invariant "
        "L1 loss: we compute the loss for both possible assignments and take the minimum. "
        "This lets the model freely assign either source to either output head.")

# ─────────────────────────────────────────────
# SLIDE 6: Results
# ─────────────────────────────────────────────
def s6_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    _tb(slide, 0.5, 0.25, 9.0, 0.6, "Results & Findings", sz=30, bold=True, color=ACCENT)

    # Left: metrics cards
    # Card 1: Pipeline validation
    _card(slide, 0.3, 1.0, 4.8, 1.0)
    tf = _tb(slide, 0.5, 1.05, 4.5, 0.9, "Pipeline Validation", sz=14, bold=True, color=ACCENT2)
    _bp(tf, "✓  Dataset diagnostics passed — tensors RGB, float32, [0, 1]", sz=12, color=WHITE)
    _bp(tf, "✓  End-to-end training + evaluation pipeline functional", sz=12, color=WHITE)

    # Card 2: Overfit
    _card(slide, 0.3, 2.2, 4.8, 1.5)
    tf = _tb(slide, 0.5, 2.25, 4.5, 1.4, "Overfit Sanity Check  (100 epochs · 16 samples · base_ch=32)", sz=13, bold=True, color=ORANGE)
    _bp(tf, "Train L1:   0.467  →  0.174", sz=13, color=WHITE)
    _bp(tf, "Best Val L1:   0.130", sz=13, color=WHITE)
    _bp(tf, "Model memorises small dataset → confirms pipeline correctness", sz=12, color=LGRAY)

    # Card 3: Baseline
    _card(slide, 0.3, 3.9, 4.8, 1.5)
    tf = _tb(slide, 0.5, 3.95, 4.5, 1.4, "Quick Baseline  (5 epochs · base_ch=16 · full data)", sz=13, bold=True, color=ORANGE)
    _bp(tf, "Val L1:  0.421   |   PSNR:  12.24 dB   |   SSIM:  0.286", sz=13, color=WHITE)
    _bp(tf, "The baseline produces separated predictions showing early layer separation.", sz=12, color=WHITE)
    _bp(tf, "Results are preliminary — stronger training is planned.", sz=12, color=LGRAY)

    # Right: images
    _try_image(slide,
        "docs/presentation/assets/overfit_loss_curve.png",
        5.3, 0.9, 4.4, 2.4,
        "[Insert actual result here]\nexperiments/overfit_test/loss_curve.png")
    _try_image(slide,
        "docs/presentation/assets/overfit_epoch_100_samples.png",
        5.3, 3.5, 4.4, 2.3,
        "[Insert actual result here]\nexperiments/overfit_test/visuals/epoch_100_samples.png")

    _tb(slide, 5.3, 5.85, 4.4, 0.3, "Epoch 100 predictions  (S1 · S2 · Mix · P1 · P2 · Err1 · Err2)", sz=9, italic=True, color=LGRAY, align=PP_ALIGN.CENTER)

    _notes(slide,
        "We ran a 100-epoch overfit sanity check on 16 samples with base_channels=32. "
        "The loss dropped from 0.467 to 0.174 and the best validation L1 reached 0.130. "
        "The epoch 100 visual shows the model learning to separate structure and color. "
        "We also ran a quick 5-epoch baseline on the full dataset with base_channels=16 "
        "which achieved 12.24 dB PSNR and 0.286 SSIM. These are preliminary results — "
        "the model learns meaningful structure but output quality remains preliminary. "
        "The framework is ready for stronger training.")

# ─────────────────────────────────────────────
# SLIDE 7: Next Steps & Conclusion
# ─────────────────────────────────────────────
def s7_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); _bg(slide)
    _tb(slide, 0.5, 0.25, 9.0, 0.6, "Next Steps & Conclusion", sz=30, bold=True, color=ACCENT)

    # Next steps card
    _card(slide, 0.3, 1.0, 4.8, 3.5)
    tf = _tb(slide, 0.5, 1.1, 4.5, 3.3, "Planned Improvements", sz=17, bold=True, color=ORANGE)
    _bp(tf, "Scale to base_channels = 32 → 64", sz=14, color=WHITE)
    _bp(tf, "Train for 30+ epochs on larger dataset", sz=14, color=WHITE)
    _bp(tf, "Scale to 1,000–5,000 synthetic pairs", sz=14, color=WHITE)
    _bp(tf, "Add reconstruction-consistency loss", sz=14, color=WHITE)
    _bp(tf, "   α · p₁ + (1−α) · p₂  ≈  mixture", sz=12, color=LGRAY)
    _bp(tf, "Evaluate by alpha range and synthesis mode", sz=14, color=WHITE)
    _bp(tf, "Compare harder blending: blur, mask, γ, shift", sz=14, color=WHITE)

    # Conclusion card
    _card(slide, 5.3, 1.0, 4.4, 2.0)
    tf = _tb(slide, 5.5, 1.1, 4.1, 1.8, "Conclusion", sz=17, bold=True, color=ACCENT2)
    _bp(tf, "We implemented an end-to-end separation framework for double-exposure images.", sz=13, color=WHITE)
    _bp(tf, "The baseline successfully produces separated output predictions.", sz=13, color=WHITE)
    _bp(tf, "The outputs show early layer separation behavior.", sz=13, color=WHITE)

    # Key contribution card
    _card(slide, 5.3, 3.2, 4.4, 1.3, color=RGBColor(0x1C, 0x2D, 0x3A), border=ACCENT)
    tf = _tb(slide, 5.5, 3.3, 4.1, 1.1, "Key Contribution", sz=15, bold=True, color=ACCENT)
    _bp(tf, "A controlled synthetic framework for studying double-exposure separation and analysing which visual factors make separation difficult.", sz=13, color=WHITE)

    # Thank you
    _card(slide, 0.3, 4.8, 9.4, 0.9, color=BG_CARD)
    _tb(slide, 0.5, 4.9, 9.0, 0.7,
        "Thank you  ·  Questions?",
        sz=24, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    _notes(slide,
        "Moving forward, we will scale our training with more data and a larger model. "
        "We plan to add a reconstruction-consistency loss and evaluate against harder "
        "blending regimes. The key contribution is the controlled synthetic framework "
        "itself, which lets us precisely measure what makes separation difficult. "
        "Thank you for listening — happy to take questions.")

# ─────────────────────────────────────────────
def create_final(output_path):
    prs = Presentation()
    prs.slide_width  = Emu(12192000)  # 16:9
    prs.slide_height = Emu(6858000)

    s1_title(prs)
    s2_problem(prs)
    s3_pipeline(prs)
    s4_examples(prs)
    s5_model(prs)
    s6_results(prs)
    s7_conclusion(prs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"✓ Final presentation saved to {output_path}")
    print(f"  Slides: {len(prs.slides)}")

if __name__ == "__main__":
    out = PROJECT_ROOT / "docs" / "presentation" / "CS585_Double_Exposure_Separation_Final.pptx"
    create_final(str(out))
