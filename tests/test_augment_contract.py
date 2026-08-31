import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import prediction_row, score_directory, write_predictions
from src.augment import (
    BLUR_SIGMAS,
    CENTER_CROP,
    COLOR_JITTER,
    EVAL_PRESETS,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    apply_named,
    center_crop,
    color_jitter,
    down_up_resize,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    random_train_augment,
)
from src.config import model_path
import random


def _rgb(h=64, w=80, seed=0) -> Image.Image:
    arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class AugmentTests(unittest.TestCase):
    def setUp(self):
        self.img = _rgb()

    def test_presets_keep_size(self):
        for name, op, param in EVAL_PRESETS:
            out = apply_named(self.img, op, param)
            self.assertEqual(out.size, self.img.size, msg=name)
            self.assertEqual(out.mode, "RGB")

    def test_jpeg_official_qualities(self):
        self.assertEqual(JPEG_QUALITIES, (90, 70, 50, 30))
        for q in JPEG_QUALITIES:
            out = jpeg_compress(self.img, q)
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.size, self.img.size)

    def test_blur_official_sigmas(self):
        self.assertEqual(BLUR_SIGMAS, (0.5, 1.0, 2.0))
        for sigma in BLUR_SIGMAS:
            out = gaussian_blur(self.img, sigma)
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.size, self.img.size)

    def test_noise_official_sigmas(self):
        self.assertEqual(NOISE_SIGMAS, (0.02, 0.05, 0.10))
        for sigma in NOISE_SIGMAS:
            out = gaussian_noise(self.img, sigma)
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.size, self.img.size)

    def test_jitter_and_crop_keep_rgb_size(self):
        self.assertEqual(COLOR_JITTER, 0.20)
        self.assertEqual(CENTER_CROP, 0.80)
        jittered = color_jitter(self.img, COLOR_JITTER)
        cropped = center_crop(self.img, CENTER_CROP)
        for out in (jittered, cropped):
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.size, self.img.size)

    def test_resize_then_up_restores_spatial_size(self):
        self.assertEqual(RESIZE_SCALES, (0.5, 0.25))
        for scale in RESIZE_SCALES:
            out = down_up_resize(self.img, scale)
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.size, self.img.size)

    def test_jpeg_changes_pixels(self):
        out = jpeg_compress(self.img, 30)
        self.assertEqual(out.size, self.img.size)
        self.assertFalse(np.array_equal(np.asarray(self.img), np.asarray(out)))

    def test_random_augment(self):
        out = random_train_augment(self.img, random.Random(1))
        self.assertEqual(out.mode, "RGB")


class JsonContractTests(unittest.TestCase):
    def test_prediction_row_clamps_and_keys(self):
        low = prediction_row(Path("a.jpg"), -0.25)
        high = prediction_row("b.png", 1.4)
        self.assertEqual(set(low), {"image_path", "pred"})
        self.assertEqual(set(high), {"image_path", "pred"})
        self.assertEqual(low["pred"], 0.0)
        self.assertEqual(high["pred"], 1.0)
        self.assertEqual(low["image_path"], "a.jpg")

    def test_write_predictions_schema(self):
        rows = [prediction_row("a.jpg", 0.12), prediction_row("b.png", 0.88)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "preds.json"
            write_predictions(rows, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(loaded), 2)
        for row in loaded:
            self.assertEqual(set(row), {"image_path", "pred"})
            self.assertGreaterEqual(row["pred"], 0.0)
            self.assertLessEqual(row["pred"], 1.0)
            self.assertIsInstance(row["pred"], float)

    def test_score_directory_samples_ranking(self):
        weights = model_path()
        self.assertTrue(weights.exists(), f"missing smoke weights at {weights}")
        samples = ROOT / "samples"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "preds.json"
            rows = score_directory(samples, out, use_tta=False)
            disk = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rows, disk)
        self.assertGreaterEqual(len(rows), 2)
        by_name = {}
        for row in rows:
            self.assertEqual(set(row), {"image_path", "pred"})
            self.assertGreaterEqual(row["pred"], 0.0)
            self.assertLessEqual(row["pred"], 1.0)
            by_name[Path(row["image_path"]).name] = row["pred"]
        self.assertIn("real_noise.jpg", by_name)
        self.assertIn("synth_pattern.png", by_name)


if __name__ == "__main__":
    unittest.main()
