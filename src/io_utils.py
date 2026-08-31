from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    paths = [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return paths


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
