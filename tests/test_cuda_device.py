"""Shipped embed device must be CUDA when torch.cuda is available.

On a CPU-only machine (this dev box: Intel UHD only) the whole class is skipped
rather than failing -- the invariant is "use the GPU when there is one", which is
vacuously satisfied with no GPU. The full-scale CUDA training box will run it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.features import clip_embed_batch, get_device, load_clip
from src.io_utils import open_image


@unittest.skipUnless(torch.cuda.is_available(), "no CUDA on this machine")
class CudaDeviceTests(unittest.TestCase):
    def test_get_device_is_cuda_when_available(self):
        device = get_device()
        self.assertEqual(device.type, "cuda")
        self.assertTrue(torch.cuda.get_device_name(0))

    def test_load_clip_and_embed_run_on_cuda(self):
        _processor, model, device = load_clip()
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        sample = ROOT / "samples" / "real_noise.jpg"
        self.assertTrue(sample.exists())
        feats = clip_embed_batch([open_image(sample)])
        self.assertEqual(feats.shape[0], 1)
        self.assertGreater(feats.shape[1], 0)
