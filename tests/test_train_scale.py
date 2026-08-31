"""Ingest label-2 mapping, paired official views, TTA scoring alignment."""

from __future__ import annotations

import inspect
import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import PAIR_FAMILIES, paired_official_views
from src.data import ingest_sid_record, mapped_binary_label
from src.features import embed_one
from src.score import embed_for_score


def _rgb(seed=1) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (48, 56, 3), dtype=np.uint8), mode="RGB")


class LabelMapTests(unittest.TestCase):
    def test_sid_label_2_is_aigc_positive(self):
        self.assertEqual(mapped_binary_label(0), 0)
        self.assertEqual(mapped_binary_label(1), 1)
        self.assertEqual(mapped_binary_label(2), 1)
        ingested = ingest_sid_record({"label": 2, "image": _rgb()})
        self.assertIsNotNone(ingested)
        self.assertEqual(ingested["label"], 1)
        self.assertEqual(ingested["folder"], "aigc")
        self.assertEqual(ingested["sid_label"], 2)


class PairedViewTests(unittest.TestCase):
    def test_paired_official_views(self):
        img = _rgb()
        clean, degraded, family = paired_official_views(img, random.Random(2026))
        self.assertIn(family, PAIR_FAMILIES)
        self.assertEqual(clean.mode, "RGB")
        self.assertEqual(degraded.mode, "RGB")
        self.assertEqual(degraded.size, clean.size)
        self.assertEqual(clean.size, img.size)


class TtaAlignmentTests(unittest.TestCase):
    def test_embed_for_score_defaults_to_tta(self):
        params = inspect.signature(embed_for_score).parameters
        self.assertTrue(params["use_tta"].default is True)

    def test_default_score_embed_matches_tta_not_plain(self):
        img = _rgb(3)
        feat_default = embed_for_score(img, "openai/clip-vit-base-patch32")
        feat_tta = embed_one(img, "openai/clip-vit-base-patch32", use_tta=True)
        feat_off = embed_one(img, "openai/clip-vit-base-patch32", use_tta=False)
        np.testing.assert_allclose(feat_default, feat_tta, rtol=1e-5, atol=1e-5)
        self.assertFalse(
            np.allclose(feat_tta, feat_off, rtol=1e-3, atol=1e-3),
            "TTA-off scoring path must differ from the shipped TTA score embed",
        )
        src = (ROOT / "scripts" / "extract_features.py").read_text(encoding="utf-8")
        self.assertIn("embed_for_score", src)
        infer_src = (ROOT / "infer.py").read_text(encoding="utf-8")
        self.assertIn("embed_for_score", infer_src)
