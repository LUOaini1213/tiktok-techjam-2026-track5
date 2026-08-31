from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile, ImageOps

# A partially-downloaded / truncated file (common after a social-media repost) would
# otherwise raise mid-decode and abort the whole submission run. Tolerate it.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    paths = [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return paths


def open_image(path: Path) -> Image.Image:
    """Load an image as RGB, honoring EXIF orientation.

    Phone photos carry an EXIF orientation flag; on a transform-robustness track a
    mis-oriented input is silently wrong, so apply exif_transpose before converting.
    .convert("RGB") also normalizes P/LA/CMYK/16-bit/RGBA modes to 3-channel.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
