"""Frozen CLIP image features plus a cheap forensic vector."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from .augment import jpeg_compress, down_up_resize, to_rgb

FORENSIC_DIM = 28
CLIP_DIM = 512  # projected (visual_projection) feature width
CLIP_PRE_DIM = 768  # pre-projection (pooler_output, post-LayerNorm CLS) feature width

# Two CLIP feature variants are cached from ONE forward pass and A/B'd at train time:
#   "proj" = visual_projection(pooler_output), 512-d  (UnivFD's actual probe input)
#   "pre"  = pooler_output, 768-d                     (post-LayerNorm CLS, pre-projection)
# Cozzolino et al. (arXiv:2312.00195) find pre-projection usually beats projected for
# linear-probe fake detection; train.py picks the winner by GroupKFold CV AUC.
DINO_MODEL_ID = "facebook/dinov2-small"
DINO_DIM = 384  # DINOv2-small CLS width
FEATURE_VARIANTS = ("pre", "proj", "dino", "fuse")
BASE_DIMS = {"pre": CLIP_PRE_DIM, "proj": CLIP_DIM, "dino": DINO_DIM, "fuse": CLIP_DIM + DINO_DIM}
DEFAULT_FEATURE_VARIANT = "pre"


def clip_dim_for_variant(variant: str) -> int:
    """Width of the backbone part of a variant (before the forensic dims are appended)."""
    if variant not in BASE_DIMS:
        raise ValueError(f"unknown feature variant {variant!r}; expected one of {FEATURE_VARIANTS}")
    return BASE_DIMS[variant]


def feature_dim_for_variant(variant: str, use_forensic: bool) -> int:
    return clip_dim_for_variant(variant) + (FORENSIC_DIM if use_forensic else 0)


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


def clip_embed_batch_dual(
    images: Sequence[Image.Image],
    model_id: str = "openai/clip-vit-base-patch32",
) -> dict[str, np.ndarray]:
    """One CLIP forward -> both L2-normalized variants: {"pre": 768-d, "proj": 512-d}.

    pooler_output = post-LayerNorm CLS token = PRE-projection feature (768-d).
    visual_projection(pooler_output) = projected feature (512-d). Both are captured
    from the SAME forward so there is no extra cost to A/B them.
    """
    processor, model, device = load_clip(model_id)
    if device.type != "cuda" and torch.cuda.is_available():
        raise RuntimeError("CUDA is available but CLIP embed is not on GPU")
    images = [to_rgb(im) for im in images]
    inputs = processor(images=list(images), return_tensors="pt")
    pixel = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision = model.vision_model(pixel_values=pixel)
        pooled = vision.pooler_output
        projected = model.visual_projection(pooled)
        pre = torch.nn.functional.normalize(pooled.float(), dim=-1)
        proj = torch.nn.functional.normalize(projected.float(), dim=-1)
    return {
        "pre": pre.cpu().numpy().astype(np.float32),
        "proj": proj.cpu().numpy().astype(np.float32),
    }


def clip_embed_batch(
    images: Sequence[Image.Image],
    model_id: str = "openai/clip-vit-base-patch32",
    variant: str = "proj",
) -> np.ndarray:
    """Back-compat single-variant helper (defaults to the 512-d projected feature)."""
    return clip_embed_batch_dual(images, model_id=model_id)[variant]


@lru_cache(maxsize=1)
def load_dino(model_id: str = DINO_MODEL_ID):
    device = get_device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return processor, model, device


def dino_embed_batch(images: Sequence[Image.Image], model_id: str = DINO_MODEL_ID) -> np.ndarray:
    """DINOv2 CLS token (pooler_output), L2-normalised, shape (n, DINO_DIM).

    Self-supervised features fail on different images than CLIP's language-aligned ones;
    a linear head over both is the cheapest fusion that exploits that.
    """
    processor, model, device = load_dino(model_id)
    images = [to_rgb(im) for im in images]
    inputs = processor(images=list(images), return_tensors="pt")
    pixel = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values=pixel)
        cls = torch.nn.functional.normalize(out.pooler_output.float(), dim=-1)
    return cls.cpu().numpy().astype(np.float32)


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


def _tta_clip_means(
    image: Image.Image,
    model_id: str,
    use_tta: bool,
    tta_jpeg_quality: int,
    tta_resize_scale: float,
) -> dict[str, np.ndarray]:
    """TTA-averaged, re-L2-normalized CLIP means for BOTH variants from one view set."""
    views = [image]
    if use_tta:
        views.append(jpeg_compress(image, tta_jpeg_quality))
        views.append(down_up_resize(image, tta_resize_scale))
    dual = clip_embed_batch_dual(views, model_id=model_id)
    means = {}
    for variant, feats in dual.items():
        mean = feats.mean(axis=0)
        # Re-L2-normalize after averaging: mean-of-unit-vectors has norm < 1 (it encodes
        # view agreement), which otherwise couples the feature scale to the TTA view count.
        means[variant] = (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)
    return means


def embed_one_dual(
    image: Image.Image,
    model_id: str,
    use_forensic: bool = True,
    use_tta: bool = True,
    tta_jpeg_quality: int = 70,
    tta_resize_scale: float = 0.5,
    variants: "Sequence[str] | None" = None,
) -> dict[str, np.ndarray]:
    """Fused feature for EVERY variant: pre / proj / dino / fuse, each + forensic.

    The forensic vector is computed ONCE and shared; only the CLIP half differs. This is
    the single shared embed path -- extract/train/eval/infer all go through here so the
    two cached variants are guaranteed identical to what infer produces at score time.
    """
    image = to_rgb(image)
    wanted = tuple(variants) if variants else FEATURE_VARIANTS
    means = _tta_clip_means(image, model_id, use_tta, tta_jpeg_quality, tta_resize_scale)
    # DINOv2 is only run when a requested variant needs it, so a head shipped on a pure
    # CLIP variant keeps its original per-image cost. Clean view only: no TTA for the
    # expensive tower.
    dino = dino_embed_batch([image])[0] if any(v in ("dino", "fuse") for v in wanted) else None
    forensic = forensic_vector(image) if use_forensic else np.zeros(FORENSIC_DIM, dtype=np.float32)
    bases = {"pre": means["pre"], "proj": means["proj"]}
    if dino is not None:
        bases["dino"] = dino
        bases["fuse"] = np.concatenate([means["proj"], dino]).astype(np.float32)
    return {v: fuse_features(bases[v], forensic, use_forensic) for v in wanted}


def embed_one(
    image: Image.Image,
    model_id: str,
    use_forensic: bool = True,
    use_tta: bool = True,
    tta_jpeg_quality: int = 70,
    tta_resize_scale: float = 0.5,
    variant: str = DEFAULT_FEATURE_VARIANT,
) -> np.ndarray:
    """Single-variant fused feature (default pre-projection). Shares embed_one_dual."""
    return embed_one_dual(
        image,
        model_id=model_id,
        use_forensic=use_forensic,
        use_tta=use_tta,
        tta_jpeg_quality=tta_jpeg_quality,
        tta_resize_scale=tta_resize_scale,
        variants=(variant,),
    )[variant]


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
