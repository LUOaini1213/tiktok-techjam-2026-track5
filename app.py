"""Gradio demo: score an image and replay social-media transforms."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.augment import center_crop, gaussian_blur, gaussian_noise, jpeg_compress
from src.config import load_config, model_path
from src.features import DEFAULT_FEATURE_VARIANT, embed_one
from src.model import load_bundle, predict_proba

cfg = load_config()
BUNDLE = None
META = {}


def load_weights():
    global BUNDLE, META
    BUNDLE = load_bundle(model_path(cfg))
    META = BUNDLE.get("meta", {})


def score_image(
    image: Image.Image, jpeg_q: int, blur_sigma: float, noise_sigma: float, crop_ratio: float
) -> tuple:
    if image is None:
        return "Upload an image", 0.0, image
    if BUNDLE is None:
        load_weights()
    work = image.convert("RGB")
    if jpeg_q < 100:
        work = jpeg_compress(work, int(jpeg_q))
    if blur_sigma > 0:
        work = gaussian_blur(work, float(blur_sigma))
    if noise_sigma > 0:
        work = gaussian_noise(work, float(noise_sigma))
    if crop_ratio < 1.0:
        work = center_crop(work, float(crop_ratio))
    feat = embed_one(
        work,
        model_id=META.get("clip_model_id", cfg["clip_model_id"]),
        use_forensic=META.get("use_forensic", True),
        use_tta=META.get("use_tta", True),
        variant=META.get("feature_variant", DEFAULT_FEATURE_VARIANT),
    )
    pred = float(predict_proba(BUNDLE, feat.reshape(1, -1))[0])
    pred = min(1.0, max(0.0, pred))
    label = "Likely AIGC" if pred >= 0.5 else "Likely authentic"
    return f"{label}  (p_AIGC = {pred:.3f})", pred, work


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
        ],
        outputs=[
            gr.Textbox(label="Verdict"),
            gr.Number(label="p(AIGC)"),
            gr.Image(type="pil", label="Transformed view"),
        ],
        title="RepostGuard — AIGC detection after redistribution",
        description=(
            "TikTok TechJam 2026 Track 5. Frozen CLIP ViT-B/32 + DINOv2-small + forensic stats + TTA. "
            f"{status} Drag JPEG/blur/crop to simulate social-media reposts."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
