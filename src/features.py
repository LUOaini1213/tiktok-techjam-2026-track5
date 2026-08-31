"""Frozen CLIP image features plus a cheap forensic vector."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct
from transformers import CLIPModel, CLIPProcessor

from .augment import jpeg_compress, down_up_resize, to_rgb

FORENSIC_DIM = 28
CLIP_DIM = 512

# The forensic branch runs on a NATIVE-scale center crop of this size, never a
# bilinear downscale: cropping preserves the high-frequency GAN-upsampling combs and
# JPEG 8x8 grid that the residual/DCT/FFT stats exist to measure, whereas resizing to
# 128 low-pass-filters them away (aliasing past the lowered Nyquist).
FORENSIC_WORK_SIZE = 256


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_clip(model_id: str = "openai/clip-vit-base-patch32"):
    device = get_device()
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return processor, model, device


def clip_embed_batch(
    images: Sequence[Image.Image],
    model_id: str = "openai/clip-vit-base-patch32",
) -> np.ndarray:
    processor, model, device = load_clip(model_id)
    if device.type != "cuda" and torch.cuda.is_available():
        raise RuntimeError("CUDA is available but CLIP embed is not on GPU")
    images = [to_rgb(im) for im in images]
    inputs = processor(images=list(images), return_tensors="pt")
    pixel = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision = model.vision_model(pixel_values=pixel)
        pooled = vision.pooler_output
        feats = model.visual_projection(pooled)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def _forensic_work(image: Image.Image, size: int = FORENSIC_WORK_SIZE) -> np.ndarray:
    """A size x size NATIVE-scale center crop (RGB, float32 in [0, 255]).

    Center-crop (no resample) keeps native pixel frequencies intact; small images are
    symmetric-padded up to `size` rather than upscaled. This replaces the old 128x128
    bilinear downscale that destroyed the forensic signal.
    """
    arr = np.asarray(to_rgb(image), dtype=np.float32)
    h, w = arr.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    crop = arr[top : top + size, left : left + size]
    ch, cw = crop.shape[:2]
    if ch < size or cw < size:
        crop = np.pad(crop, ((0, size - ch), (0, size - cw), (0, 0)), mode="symmetric")
    return crop


def _luma(rgb01: np.ndarray) -> np.ndarray:
    return (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation guarded against a constant channel (returns 0.0)."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def forensic_vector(image: Image.Image) -> np.ndarray:
    """NPR-residual / block-DCT / FFT-ring / color stats computed at NATIVE resolution.

    28-D. Complements CLIP by capturing high-frequency synthesis fingerprints that CLIP
    loses and that a 128x128 downscale would have aliased away.
    """
    rgb01 = _forensic_work(image) / 255.0
    gray = _luma(rgb01)
    n = gray.shape[0]  # == FORENSIC_WORK_SIZE

    # --- NPR-style residual (native scale) ---
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    resid = gray - blur
    npr_stats = np.array(
        [
            resid.mean(),
            resid.std(),
            np.abs(resid).mean(),
            np.percentile(resid, 5),
            np.percentile(resid, 95),
        ],
        dtype=np.float32,
    )

    # --- 8x8 block DCT high/low energy ratio: 8x8 blocks now align with the JPEG grid ---
    block = 8
    energies = []
    for y in range(0, n, block):
        for x in range(0, n, block):
            patch = gray[y : y + block, x : x + block]
            coeff = dct(dct(patch, axis=0, norm="ortho"), axis=1, norm="ortho")
            low = np.abs(coeff[:2, :2]).mean()
            high = np.abs(coeff[2:, 2:]).mean()
            energies.append(high / (low + 1e-6))
    energies = np.array(energies, dtype=np.float32)
    dct_stats = np.array(
        [
            energies.mean(),
            energies.std(),
            np.percentile(energies, 25),
            np.percentile(energies, 75),
            np.percentile(energies, 95),
        ],
        dtype=np.float32,
    )

    # --- FFT radial rings as fractions of Nyquist; the top rings (>=0.75) hold the
    #     GAN/upsampling spectral peaks that downscaling would have removed ---
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    cy = cx = n // 2
    half = n / 2.0
    yy, xx = np.ogrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rings = []
    for lo, hi in ((0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5)):
        mask = (r >= lo * half) & (r < hi * half)
        rings.append(mag[mask].mean() if mask.any() else 0.0)
    fft_stats = np.array(rings, dtype=np.float32)

    # --- color stats (resolution-agnostic) ---
    color_stats = np.array(
        [
            *rgb01.mean(axis=(0, 1)),
            *rgb01.std(axis=(0, 1)),
            _safe_corr(rgb01[..., 0], rgb01[..., 1]),
            _safe_corr(rgb01[..., 1], rgb01[..., 2]),
            _safe_corr(rgb01[..., 0], rgb01[..., 2]),
        ],
        dtype=np.float32,
    )

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    extra = np.array([lap.var(), gray.mean(), gray.std()], dtype=np.float32)

    vec = np.concatenate([npr_stats, dct_stats, fft_stats, color_stats, extra])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert vec.shape[0] == FORENSIC_DIM, f"forensic dim {vec.shape[0]} != {FORENSIC_DIM}"
    return vec


def fuse_features(clip_vec: np.ndarray, forensic_vec: np.ndarray, use_forensic: bool) -> np.ndarray:
    if not use_forensic:
        return clip_vec.astype(np.float32)
    return np.concatenate([clip_vec, forensic_vec], axis=-1).astype(np.float32)


def embed_one(
    image: Image.Image,
    model_id: str,
    use_forensic: bool = True,
    use_tta: bool = True,
    tta_jpeg_quality: int = 70,
    tta_resize_scale: float = 0.5,
) -> np.ndarray:
    image = to_rgb(image)
    views = [image]
    if use_tta:
        views.append(jpeg_compress(image, tta_jpeg_quality))
        views.append(down_up_resize(image, tta_resize_scale))
    clip_feats = clip_embed_batch(views, model_id=model_id)
    clip_mean = clip_feats.mean(axis=0)
    # Re-L2-normalize after averaging: mean-of-unit-vectors has norm < 1 (it encodes
    # view agreement), which otherwise couples the feature scale to the TTA view count.
    clip_mean = clip_mean / (np.linalg.norm(clip_mean) + 1e-8)
    forensic = forensic_vector(image) if use_forensic else np.zeros(FORENSIC_DIM, dtype=np.float32)
    return fuse_features(clip_mean, forensic, use_forensic)


def embed_many(
    images: Iterable[Image.Image],
    model_id: str,
    use_forensic: bool,
    batch_size: int = 8,
) -> np.ndarray:
    images = [to_rgb(im) for im in images]
    clips = []
    for i in range(0, len(images), batch_size):
        clips.append(clip_embed_batch(images[i : i + batch_size], model_id=model_id))
    clip_mat = np.concatenate(clips, axis=0) if clips else np.zeros((0, CLIP_DIM), dtype=np.float32)
    if not use_forensic:
        return clip_mat
    forensic = np.stack([forensic_vector(im) for im in images], axis=0)
    return np.concatenate([clip_mat, forensic], axis=1)
