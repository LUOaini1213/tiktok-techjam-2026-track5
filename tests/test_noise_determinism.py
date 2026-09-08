"""Evaluation-time Gaussian noise must be reproducible: same image -> same noise draw.

Before content-seeding, gaussian_noise() drew from an unseeded generator, so the three
noise rows of the robustness table drifted in the third decimal on every re-run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import apply_named, gaussian_noise


def _img(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (64, 80, 3), dtype=np.uint8), "RGB")


class NoiseDeterminismTests(unittest.TestCase):
    def test_same_image_same_noise(self):
        a = np.asarray(gaussian_noise(_img(1), 0.10))
        b = np.asarray(gaussian_noise(_img(1), 0.10))
        self.assertTrue(np.array_equal(a, b), "noise draw must be a function of image content")

    def test_apply_named_path_is_deterministic_too(self):
        a = np.asarray(apply_named(_img(2), "noise", 0.05))
        b = np.asarray(apply_named(_img(2), "noise", 0.05))
        self.assertTrue(np.array_equal(a, b))

    def test_different_images_get_different_noise(self):
        base1, base2 = np.asarray(_img(3)).astype(int), np.asarray(_img(4)).astype(int)
        n1 = np.asarray(gaussian_noise(_img(3), 0.10)).astype(int) - base1
        n2 = np.asarray(gaussian_noise(_img(4), 0.10)).astype(int) - base2
        self.assertFalse(np.array_equal(n1, n2), "content seeding must not collapse to one draw")

    def test_explicit_rng_still_honoured(self):
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        a = np.asarray(gaussian_noise(_img(5), 0.10, rng1))
        b = np.asarray(gaussian_noise(_img(5), 0.10, rng2))
        self.assertTrue(np.array_equal(a, b))

    def test_noise_actually_changes_the_image(self):
        base = np.asarray(_img(6)).astype(int)
        noisy = np.asarray(gaussian_noise(_img(6), 0.10)).astype(int)
        self.assertGreater(np.abs(noisy - base).mean(), 5.0)


if __name__ == "__main__":
    unittest.main()
