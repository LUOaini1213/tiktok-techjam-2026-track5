"""multi_crops geometry: counts, sizes, distinct regions, and the n=0 no-op."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import multi_crops


def _gradient(w=120, h=90) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([xx * 2 % 256, yy * 2 % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


class MultiCropTests(unittest.TestCase):
    def test_zero_is_noop(self):
        self.assertEqual(multi_crops(_gradient(), 0), [])

    def test_counts_and_sizes(self):
        img = _gradient()
        for n, expect in ((1, 1), (2, 2), (4, 4), (5, 5), (9, 5)):
            crops = multi_crops(img, n)
            self.assertEqual(len(crops), expect, n)
            for c in crops:
                self.assertEqual(c.size, img.size)
                self.assertEqual(c.mode, "RGB")

    def test_corner_crops_cover_different_regions(self):
        crops = [np.asarray(c).astype(int) for c in multi_crops(_gradient(), 4)]
        # top-left vs bottom-right corner crops of a gradient must differ substantially
        self.assertGreater(np.abs(crops[0] - crops[3]).mean(), 20)

    def test_centre_crop_is_zoomed_in(self):
        img = _gradient()
        centre = np.asarray(multi_crops(img, 1)[0]).astype(int)
        full = np.asarray(img).astype(int)
        # the centre crop resized back is not the identity (it is a zoom)
        self.assertGreater(np.abs(centre - full).mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
