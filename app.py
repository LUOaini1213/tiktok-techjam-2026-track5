"""Gradio demo: score an image and replay social-media transforms."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.augment import center_crop, gaussian_blur, gaussian_noise, jpeg_compress
from src.config import load_config, model_path
from src.decision import verdict
from src.features import DEFAULT_FEATURE_VARIANT
from src.model import load_bundle
from src.repost import CHAINS, apply_chain, rescale
from src.score import embed_for_score, score_from_embed

cfg = load_config()
BUNDLE = None
META = {}


def load_weights():
    global BUNDLE, META
    if not torch.cuda.is_available():
        torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    BUNDLE = load_bundle(model_path(cfg))
    META = BUNDLE.get("meta", {})


def score_image(
    image: Image.Image, jpeg_q: int, blur_sigma: float, noise_sigma: float, crop_ratio: float,
    resize_scale: float = 1.0, scenario: str = "Manual sliders",
) -> tuple:
    if image is None:
        return "Upload an image", 0.0, image
    if BUNDLE is None:
        load_weights()
    original = ImageOps.exif_transpose(image).convert("RGB")
    work = original
    if jpeg_q < 100:
        work = jpeg_compress(work, int(jpeg_q))
    if blur_sigma > 0:
        work = gaussian_blur(work, float(blur_sigma))
    if noise_sigma > 0:
        work = gaussian_noise(work, float(noise_sigma))
    if crop_ratio < 1.0:
        work = center_crop(work, float(crop_ratio))
    if resize_scale < 1.0:
        work = rescale(work, float(resize_scale))
    if scenario in CHAINS:
        # A selected scenario is applied directly to the uploaded original, so slider
        # settings never silently stack another transformation on a named scenario.
        work = apply_chain(original, scenario)
    feat = embed_for_score(
        work,
        model_id=META.get("clip_model_id", cfg["clip_model_id"]),
        use_forensic=META.get("use_forensic", True),
        use_tta=META.get("use_tta", True),
        variant=META.get("feature_variant", DEFAULT_FEATURE_VARIANT),
        tta_jpeg_quality=META.get("tta_jpeg_quality", cfg["tta_jpeg_quality"]),
        tta_resize_scale=META.get("tta_resize_scale", cfg["tta_resize_scale"]),
    )
    try:
        pred = score_from_embed(BUNDLE, feat)
    except ValueError:
        return "Needs review (invalid model score; no authenticity claim)", 0.5, work
    policy = META.get("decision_policy")
    label = verdict(pred, policy)
    detail = (f"AI flag ≥ {policy['aigc_threshold']:.3f}; low signal ≤ {policy['authentic_threshold']:.3f}. "
              if policy else "Legacy binary policy. ")
    return (f"{label}  (AIGC score = {pred:.3f})\n{detail}"
            "Thresholds are calibration-specific. Low signal is not proof of authenticity."), pred, work


def main() -> None:
    try:
        load_weights()
        status = "Weights loaded."
    except FileNotFoundError as e:
        status = str(e)

    demo = gr.Interface(
        fn=score_image,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Slider(30, 100, value=100, step=1, label="JPEG quality (100 = skip)"),
            gr.Slider(0, 2.0, value=0, step=0.1, label="Gaussian blur σ"),
            gr.Slider(0, 0.10, value=0, step=0.01, label="Gaussian noise σ (0 = skip)"),
            gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="Center crop ratio"),
            gr.Slider(0.125, 1.0, value=1.0, step=0.125, label="Resize down/up scale"),
            gr.Dropdown(["Manual sliders", *CHAINS], value="Manual sliders", label="Repost scenario (overrides sliders)"),
        ],
        outputs=[
            gr.Textbox(label="Verdict"),
            gr.Number(label="Calibrated AIGC score"),
            gr.Image(type="pil", label="Transformed view"),
        ],
        title="RepostGuard — AIGC detection after redistribution",
        description=(
            "Research prototype based on the TikTok TechJam 2026 Track 5 brief; never submitted. "
            "Frozen CLIP ViT-B/32 + DINOv2-small + forensic stats + TTA. "
            f"{status} Simulate single edits or a complete repost chain. "
            "Needs review means the model abstains; a low score is not proof of authenticity."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
