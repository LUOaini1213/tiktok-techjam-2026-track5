"""Build a licensed local train/val set larger than samples/ if SID-Set is unreachable.

Procedural RGB images (we own them). Not WildFake. Used only as a GPU-training fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import enable_utf8_stdout
from src.data import LABEL_AIGC, LABEL_REAL, save_split, write_image


def _natural(rng: np.random.Generator, size: int = 256) -> Image.Image:
    base = rng.normal(120, 40, (size, size, 3))
    yy, xx = np.mgrid[0:size, 0:size]
    gradient = (xx + yy)[:, :, None] * 0.15
    noise = rng.random((size, size, 3)) * 25
    arr = np.clip(base + gradient + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=0.6))


def _synthetic(rng: np.random.Generator, size: int = 256, k: int = 0) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    arr[..., 0] = (np.sin((xx + k) / 7.0) * 127 + 128).astype(np.uint8)
    arr[..., 1] = (np.cos((yy - k) / 9.0) * 127 + 128).astype(np.uint8)
    arr[..., 2] = (((xx * 3 + yy * 2 + k * 11) % 255)).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(4):
        x0, y0 = int(rng.integers(0, size - 40)), int(rng.integers(0, size - 40))
        x1, y1 = x0 + int(rng.integers(20, 80)), y0 + int(rng.integers(20, 80))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        draw.ellipse([x0, y0, x1, y1], fill=color)
    return img


def build(out_dir: Path, train_n: int, val_n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    counts = {"train": {LABEL_REAL: 0, LABEL_AIGC: 0}, "val": {LABEL_REAL: 0, LABEL_AIGC: 0}}
    for split, n in (("train", train_n), ("val", val_n)):
        for i in range(n):
            write_image(out_dir / split / "real" / f"0_{i:06d}.jpg", _natural(rng))
            write_image(out_dir / split / "aigc" / f"1_{i:06d}.jpg", _synthetic(rng, k=i))
            counts[split][LABEL_REAL] += 1
            counts[split][LABEL_AIGC] += 1
    manifest = {
        "seed": seed,
        "train_per_class": train_n,
        "val_per_class": val_n,
        "counts": counts,
        "source": "procedural_local_fallback",
        "note": "Not SID-Set. Not WildFake. Used because SID streaming was unavailable.",
    }
    save_split(out_dir / "split.json", manifest)
    return manifest


def main() -> None:
    enable_utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--train_per_class", type=int, default=64)
    p.add_argument("--val_per_class", type=int, default=16)
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config()
    out = args.out_dir or cfg["data_dir"]
    man = build(out, args.train_per_class, args.val_per_class, cfg["seed"])
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
