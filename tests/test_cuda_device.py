"""Shipped embed device must be CUDA when torch.cuda is available."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.features import clip_embed_batch, get_device, load_clip
from src.io_utils import open_image


class CudaDeviceTests(unittest.TestCase):
    def test_get_device_is_cuda_when_available(self):
        self.assertTrue(
            torch.cuda.is_available(),
            "CUDA PyTorch wheel is required; torch.cuda.is_available() is False",
        )
        device = get_device()
        self.assertEqual(device.type, "cuda")
        self.assertTrue(torch.cuda.get_device_name(0))

    def test_load_clip_and_embed_run_on_cuda(self):
        self.assertTrue(torch.cuda.is_available())
        _processor, model, device = load_clip()
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        sample = ROOT / "samples" / "real_noise.jpg"
        self.assertTrue(sample.exists())
        feats = clip_embed_batch([open_image(sample)])
        self.assertEqual(feats.shape[0], 1)
        self.assertGreater(feats.shape[1], 0)
