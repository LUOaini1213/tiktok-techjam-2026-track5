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

FORENSIC_DIM = 48
CLIP_DIM = 512


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


def _resize_gray(image: Image.Image, size: int = 128) -> np.ndarray:
    arr = np.asarray(to_rgb(image).resize((size, size), Image.BILINEAR))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def forensic_vector(image: Image.Image) -> np.ndarray:
    """Fixed-length DCT / NPR / color stats. Complements CLIP under JPEG/resize."""
    rgb = np.asarray(to_rgb(image).resize((128, 128), Image.BILINEAR), dtype=np.float32)
    rgb01 = rgb / 255.0
    gray = _resize_gray(image, 128)

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

    block = 8
    energies = []
    for y in range(0, 128, block):
        for x in range(0, 128, block):
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
        ],
        dtype=np.float32,
    )

    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    cy, cx = 64, 64
    rings = []
    yy, xx = np.ogrid[:128, :128]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 64)):
        mask = (r >= lo) & (r < hi)
        rings.append(mag[mask].mean() if mask.any() else 0.0)
    fft_stats = np.array(rings, dtype=np.float32)

    color_stats = np.concatenate(
        [
            rgb01.mean(axis=(0, 1)),
            rgb01.std(axis=(0, 1)),
            np.corrcoef(rgb01[..., 0].ravel(), rgb01[..., 1].ravel())[:1, 1],
            np.corrcoef(rgb01[..., 1].ravel(), rgb01[..., 2].ravel())[:1, 1],
            np.corrcoef(rgb01[..., 0].ravel(), rgb01[..., 2].ravel())[:1, 1],
        ]
    ).astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    extra = np.array([lap.var(), gray.mean(), gray.std()], dtype=np.float32)

    vec = np.concatenate([npr_stats, dct_stats, fft_stats, color_stats, extra])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if vec.shape[0] < FORENSIC_DIM:
        vec = np.pad(vec, (0, FORENSIC_DIM - vec.shape[0]))
    else:
        vec = vec[:FORENSIC_DIM]
    return vec.astype(np.float32)


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
