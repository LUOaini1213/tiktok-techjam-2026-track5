"""Official Track 5 transforms and training-time random augmentations."""

from __future__ import annotations

import io
import random
import zlib
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER = 0.20
CENTER_CROP = 0.80


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    image = to_rgb(image)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(to_rgb(image))
    sigma = float(sigma)
    radius = int(max(1, round(sigma * 3)))
    k = 2 * radius + 1
    out = cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(out)


def down_up_resize(image: Image.Image, scale: float) -> Image.Image:
    image = to_rgb(image)
    w, h = image.size
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = image.resize((nw, nh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(image: Image.Image, sigma: float, rng: np.random.Generator | None = None) -> Image.Image:
    """Additive Gaussian noise. Pass `rng` for a reproducible draw (feature caching)."""
    raw = np.ascontiguousarray(np.asarray(to_rgb(image)))
    arr = raw.astype(np.float32) / 255.0
    if rng is None:
        # Content-seeded: the same image always receives the same noise draw, so every
        # evaluation row is reproducible run-to-run without threading a seed through each
        # caller. (Before this, the three noise rows drifted in the third decimal per run.)
        rng = np.random.default_rng(zlib.crc32(raw.tobytes()))
    noisy = arr + rng.normal(0.0, float(sigma), size=arr.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8), mode="RGB")


def color_jitter(
    image: Image.Image,
    amount: float = COLOR_JITTER,
    rng: random.Random | None = None,
) -> Image.Image:
    image = to_rgb(image)
    if rng is None:
        # Eval path: apply the +20% side of brightness/contrast/saturation ±20%.
        deltas = (float(amount), float(amount), float(amount))
    else:
        deltas = (
            rng.uniform(-amount, amount),
            rng.uniform(-amount, amount),
            rng.uniform(-amount, amount),
        )
    image = ImageEnhance.Brightness(image).enhance(1.0 + deltas[0])
    image = ImageEnhance.Contrast(image).enhance(1.0 + deltas[1])
    image = ImageEnhance.Color(image).enhance(1.0 + deltas[2])
    return image


def center_crop(image: Image.Image, ratio: float = CENTER_CROP) -> Image.Image:
    image = to_rgb(image)
    w, h = image.size
    nw = max(1, int(round(w * ratio)))
    nh = max(1, int(round(h * ratio)))
    left = (w - nw) // 2
    top = (h - nh) // 2
    cropped = image.crop((left, top, left + nw, top + nh))
    return cropped.resize((w, h), Image.BILINEAR)


def multi_crops(image: Image.Image, n: int, scale: float = 0.8) -> list[Image.Image]:
    """`n` crops at `scale` of the image, resized back to the original size.

    n=1 -> centre; n=2..4 -> that many corners; n=5 -> four corners + centre. Resizing
    back keeps every downstream preprocessor (CLIP's 224 shortest-side, the native-scale
    forensic crop) exactly as it is for the full view.
    """
    if n <= 0:
        return []
    image = to_rgb(image)
    w, h = image.size
    cw, ch = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    corners = [(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)]
    centre = ((w - cw) // 2, (h - ch) // 2)
    boxes = [centre] if n == 1 else corners[: min(n, 4)] + ([centre] if n >= 5 else [])
    return [image.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR) for l, t in boxes]


TRANSFORM_TABLE = {
    "jpeg": lambda img, q: jpeg_compress(img, q),
    "blur": lambda img, s: gaussian_blur(img, s),
    "resize": lambda img, s: down_up_resize(img, s),
    "noise": lambda img, s: gaussian_noise(img, s),
    "jitter": lambda img, a: color_jitter(img, a),
    "crop": lambda img, r: center_crop(img, r),
}

EVAL_PRESETS = [
    ("clean", None, None),
    ("jpeg_90", "jpeg", 90),
    ("jpeg_70", "jpeg", 70),
    ("jpeg_50", "jpeg", 50),
    ("jpeg_30", "jpeg", 30),
    ("blur_0.5", "blur", 0.5),
    ("blur_1.0", "blur", 1.0),
    ("blur_2.0", "blur", 2.0),
    ("resize_0.5", "resize", 0.5),
    ("resize_0.25", "resize", 0.25),
    ("noise_0.02", "noise", 0.02),
    ("noise_0.05", "noise", 0.05),
    ("noise_0.10", "noise", 0.10),
    ("jitter_0.20", "jitter", 0.20),
    ("crop_0.80", "crop", 0.80),
]


# One degraded training view per family. The ablation showed the families we scored but
# never trained on (noise, jitter) were exactly the weakest rows, so every official
# family except crop (already the strongest untrained row) is now a training view.
PAIR_FAMILIES = ("jpeg", "blur", "resize", "noise", "jitter")


def apply_named(image: Image.Image, name: Optional[str], param) -> Image.Image:
    if name is None:
        return to_rgb(image)
    return TRANSFORM_TABLE[name](image, param)


def paired_official_views(
    image: Image.Image,
    rng: random.Random | None = None,
    family: str | None = None,
) -> tuple[Image.Image, Image.Image, str]:
    """Clean + one official degraded view (jpeg/blur/resize/noise/jitter), same size.

    Every random choice, including the noise draw, comes from `rng`, so a per-image
    seeded RNG reproduces the identical view on a resumed extraction.
    """
    rng = rng or random.Random(0)
    clean = to_rgb(image)
    family = family or rng.choice(PAIR_FAMILIES)
    if family not in PAIR_FAMILIES:
        raise ValueError(f"pair family must be one of {PAIR_FAMILIES}, got {family}")
    if family == "jpeg":
        degraded = jpeg_compress(clean, rng.choice(JPEG_QUALITIES))
    elif family == "blur":
        degraded = gaussian_blur(clean, rng.choice(BLUR_SIGMAS))
    elif family == "resize":
        degraded = down_up_resize(clean, rng.choice(RESIZE_SCALES))
    elif family == "noise":
        degraded = gaussian_noise(
            clean, rng.choice(NOISE_SIGMAS), np.random.default_rng(rng.getrandbits(32))
        )
    else:  # jitter: random +/- amount per channel from the same stream
        degraded = color_jitter(clean, COLOR_JITTER, rng)
    return clean, degraded, family


def multi_paired_official_views(
    image: Image.Image,
    rng: random.Random | None = None,
) -> list[tuple[Image.Image, str]]:
    """Clean plus one degraded view per official pair family (jpeg, blur, resize)."""
    rng = rng or random.Random(0)
    clean = to_rgb(image)
    views: list[tuple[Image.Image, str]] = [(clean, "clean")]
    for family in PAIR_FAMILIES:
        _c, degraded, name = paired_official_views(clean, rng, family=family)
        views.append((degraded, name))
    return views


def random_train_augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Stack 1-2 official-style transforms for robustness training."""
    image = to_rgb(image)
    ops = [
        lambda im: jpeg_compress(im, rng.choice(JPEG_QUALITIES)),
        lambda im: gaussian_blur(im, rng.choice(BLUR_SIGMAS)),
        lambda im: down_up_resize(im, rng.choice(RESIZE_SCALES)),
        lambda im: gaussian_noise(im, rng.choice(NOISE_SIGMAS)),
        lambda im: color_jitter(im, COLOR_JITTER, rng),
        lambda im: center_crop(im, CENTER_CROP),
    ]
    n = 1 if rng.random() < 0.55 else 2
    for op in rng.sample(ops, k=n):
        image = op(image)
    return image
