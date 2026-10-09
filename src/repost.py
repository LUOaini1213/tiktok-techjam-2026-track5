"""Ordered redistribution chains and matched, class-independent training views."""
from __future__ import annotations

import random
import numpy as np
from PIL import Image
from .augment import center_crop, gaussian_blur, gaussian_noise, jpeg_compress, to_rgb

CHAINS = {
    "crop_resize_jpeg": (("crop", .8), ("resize", .25), ("jpeg", 50)),
    "noise_jpeg": (("noise", .05), ("jpeg", 50)),
    "blur_resize_jpeg": (("blur", 1.0), ("resize", .5), ("jpeg", 30)),
    "double_repost": (("resize", .5), ("jpeg", 70), ("resize", .5), ("jpeg", 30)),
}


def rescale(image, scale, resample=Image.Resampling.BILINEAR):
    image = to_rgb(image)
    w, h = image.size
    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)
    return small.resize((w, h), resample)


def apply_chain(image, name):
    work = to_rgb(image)
    for op, value in CHAINS[name]:
        if op == "crop":
            work = center_crop(work, value)
        elif op == "resize":
            work = rescale(work, value)
        elif op == "jpeg":
            work = jpeg_compress(work, value)
        elif op == "noise":
            work = gaussian_noise(work, value)
        else:
            work = gaussian_blur(work, value)
    return work


def train_views(image, seed):
    """Seven views, same parameter distribution for either label; no new inference cost."""
    rng = random.Random(seed)
    interp = rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
    scale = rng.choice([.125, .25, .5, .75])
    work = center_crop(image, rng.uniform(.65, .95))
    work = jpeg_compress(rescale(work, scale, interp), rng.choice([30, 50, 70, 90]))
    return [("clean", to_rgb(image)), ("resize_0.25", rescale(image, .25, interp)),
            ("jpeg_30", jpeg_compress(image, 30)), ("random_crop_scale_jpeg", work),
            *[(name, apply_chain(image, name)) for name in ("noise_jpeg", "blur_resize_jpeg", "double_repost")]]
