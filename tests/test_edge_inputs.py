"""Repost-mangled input formats must never crash infer or drop a row.

Covers the exact failure modes a social-media redistribution pipeline produces:
truncated download, RGBA/CMYK color modes, EXIF-rotated phone photo, webp re-encode.
Contract: one row per input file, every pred in [0, 1], valid JSON on disk.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import score_directory
from src.config import model_path
from src.io_utils import open_image


def _base_array(seed: int = 7, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _write_edge_files(folder: Path) -> list[str]:
    """Create the five edge-case files; returns their names."""
    arr = _base_array()
    img = Image.fromarray(arr)

    # 1. Truncated JPEG: write only the first half of a valid jpeg's bytes.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    (folder / "truncated.jpg").write_bytes(data[: len(data) // 2])

    # 2. RGBA PNG.
    rgba = Image.fromarray(arr).convert("RGBA")
    rgba.putalpha(128)
    rgba.save(folder / "rgba.png")

    # 3. CMYK JPEG.
    Image.fromarray(arr).convert("CMYK").save(folder / "cmyk.jpg", format="JPEG")

    # 4. EXIF-rotated JPEG (orientation=6 -> 90 degrees CW on display).
    exif = Image.Exif()
    exif[274] = 6  # 274 = Orientation tag
    img.save(folder / "exif_rot.jpg", format="JPEG", exif=exif)

    # 5. WebP.
    img.save(folder / "photo.webp", format="WEBP")

    return ["cmyk.jpg", "exif_rot.jpg", "photo.webp", "rgba.png", "truncated.jpg"]


class EdgeInputTests(unittest.TestCase):
    def test_open_image_handles_edge_modes(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            _write_edge_files(folder)
            for name in ("rgba.png", "cmyk.jpg", "photo.webp", "exif_rot.jpg", "truncated.jpg"):
                im = open_image(folder / name)
                self.assertEqual(im.mode, "RGB", msg=name)

    def test_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            # A wide image with orientation=6 must open as tall (transposed).
            wide = Image.fromarray(_base_array()[:32, :96])
            exif = Image.Exif()
            exif[274] = 6
            wide.save(folder / "wide_rot.jpg", format="JPEG", exif=exif)
            im = open_image(folder / "wide_rot.jpg")
            self.assertEqual((im.width, im.height), (32, 96))

    def test_score_directory_edge_files_full_contract(self):
        self.assertTrue(model_path().exists(), "run scripts/smoke_train.py first")
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            names = _write_edge_files(folder)
            out = Path(td) / "preds.json"
            rows = score_directory(folder, out, use_tta=False)
            disk = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rows, disk)
        # Exactly one row per input file, in listing order.
        self.assertEqual(len(rows), len(names))
        got_names = sorted(Path(r["image_path"]).name for r in rows)
        self.assertEqual(got_names, sorted(names))
        for row in rows:
            self.assertEqual(set(row), {"image_path", "pred"})
            self.assertGreaterEqual(row["pred"], 0.0)
            self.assertLessEqual(row["pred"], 1.0)


if __name__ == "__main__":
    unittest.main()
