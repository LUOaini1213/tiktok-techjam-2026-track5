"""Error-analysis summary math: rates, FPR@95%TPR, and JPEG-30 flip counts."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.error_analysis import _fpr_at_tpr, _summarize


def _row(label: int, pred: float, pred_jpeg30: float) -> dict:
    return {"label": label, "pred": pred, "pred_jpeg30": pred_jpeg30, "path": "x"}


class ErrorSummaryTests(unittest.TestCase):
    def test_fpr_at_tpr_on_separable_scores(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
        self.assertAlmostEqual(_fpr_at_tpr(y, s, 0.95), 0.0)

    def test_fpr_at_tpr_needs_both_classes(self):
        self.assertTrue(math.isnan(_fpr_at_tpr(np.zeros(4), np.arange(4.0))))

    def test_summary_counts_rates_and_flips(self):
        scored = [
            _row(0, 0.10, 0.10),  # real, correct, stable
            _row(0, 0.20, 0.60),  # real, correct clean -> flips to FP under JPEG-30
            _row(0, 0.90, 0.90),  # real, already a FP
            _row(1, 0.95, 0.95),  # fake, correct, stable
            _row(1, 0.80, 0.30),  # fake, correct clean -> flips to FN under JPEG-30
            _row(1, 0.10, 0.10),  # fake, already a FN
        ]
        fps = [r for r in scored if r["label"] == 0 and r["pred"] >= 0.5]
        fns = [r for r in scored if r["label"] == 1 and r["pred"] < 0.5]
        s = _summarize(scored, fps, fns)

        self.assertEqual((s["n"], s["n_real"], s["n_fake"]), (6, 3, 3))
        self.assertEqual((s["n_false_positives"], s["n_false_negatives"]), (1, 1))
        self.assertAlmostEqual(s["fpr"], 1 / 3)
        self.assertAlmostEqual(s["fnr"], 1 / 3)
        self.assertEqual(s["n_flips_to_fp_jpeg30"], 1)
        self.assertEqual(s["n_flips_to_fn_jpeg30"], 1)
        # Real images drift up (+0.40/3), fakes drift down (-0.50/3) under JPEG-30.
        self.assertGreater(s["mean_score_drift_jpeg30_real"], 0)
        self.assertLess(s["mean_score_drift_jpeg30_fake"], 0)
        self.assertGreater(s["auc_clean"], s["auc_jpeg30"])


if __name__ == "__main__":
    unittest.main()
