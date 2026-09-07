"""Feature-variant contract: widths and the 6-view training augmentation. No model loads."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import PAIR_FAMILIES, multi_paired_official_views
from src.features import (
    BASE_DIMS,
    CLIP_DIM,
    DINO_DIM,
    FEATURE_VARIANTS,
    FORENSIC_DIM,
    clip_dim_for_variant,
    feature_dim_for_variant,
)


class VariantDimTests(unittest.TestCase):
    def test_all_variants_have_dims(self):
        self.assertEqual(FEATURE_VARIANTS, ("pre", "proj", "dino", "fuse"))
        for v in FEATURE_VARIANTS:
            self.assertEqual(clip_dim_for_variant(v), BASE_DIMS[v])
            self.assertEqual(feature_dim_for_variant(v, True), BASE_DIMS[v] + FORENSIC_DIM)
            self.assertEqual(feature_dim_for_variant(v, False), BASE_DIMS[v])

    def test_fuse_is_proj_plus_dino(self):
        self.assertEqual(BASE_DIMS["fuse"], CLIP_DIM + DINO_DIM)
        self.assertEqual(feature_dim_for_variant("fuse", True), 512 + 384 + 28)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            clip_dim_for_variant("vit-g")


class SixViewTests(unittest.TestCase):
    def _img(self):
        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8), "RGB")

    def test_one_view_per_family_plus_clean(self):
        views = multi_paired_official_views(self._img(), random.Random(1))
        names = [n for _, n in views]
        self.assertEqual(names[0], "clean")
        self.assertEqual(names[1:], list(PAIR_FAMILIES))
        self.assertEqual(len(views), 1 + len(PAIR_FAMILIES))
        for im, _ in views:
            self.assertEqual(im.size, (128, 96))
            self.assertEqual(im.mode, "RGB")

    def test_views_are_reproducible_from_the_seed(self):
        a = multi_paired_official_views(self._img(), random.Random("seed:train:7"))
        b = multi_paired_official_views(self._img(), random.Random("seed:train:7"))
        for (ia, _), (ib, _) in zip(a, b):
            self.assertTrue(np.array_equal(np.asarray(ia), np.asarray(ib)))

    def test_noise_and_jitter_views_differ_from_clean(self):
        views = dict((n, np.asarray(im).astype(int)) for im, n in multi_paired_official_views(self._img(), random.Random(3)))
        for fam in ("noise", "jitter"):
            self.assertGreater(np.abs(views[fam] - views["clean"]).mean(), 1.0, fam)


if __name__ == "__main__":
    unittest.main()
