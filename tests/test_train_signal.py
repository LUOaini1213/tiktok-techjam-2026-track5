"""Shipped tampered reweight + consistency rows change the fit vs unweighted LR."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train_signal import (
    expand_with_consistency,
    fit_with_training_signal,
    positive_proba,
    sid_label_from_path,
    tampered_sample_weights,
)


class TrainSignalTests(unittest.TestCase):
    def test_sid_label_from_path(self):
        self.assertEqual(sid_label_from_path("val/aigc/2_000001.jpg"), 2)
        self.assertEqual(sid_label_from_path("val/aigc/1_000001.jpg"), 1)
        self.assertEqual(sid_label_from_path("val/real/0_000001.jpg"), 0)

    def test_tampered_weights_are_higher(self):
        sid = np.array([0, 0, 1, 2, 2])
        w = tampered_sample_weights(sid, tampered_weight=5.0)
        self.assertTrue(np.all(w[sid == 2] == 5.0))
        self.assertTrue(np.all(w[sid != 2] == 1.0))

    def test_consistency_appends_mean_rows(self):
        X = np.array([[0.0, 0.0], [2.0, 2.0], [1.0, 0.0], [3.0, 0.0]])
        y = np.array([1, 1, 0, 0])
        sid = np.array([2, 2, 0, 0])
        X2, y2, s2 = expand_with_consistency(X, y, sid, views_per_image=2)
        self.assertEqual(len(X2), 6)
        np.testing.assert_allclose(X2[4], [1.0, 1.0])
        self.assertEqual(int(y2[4]), 1)
        self.assertEqual(int(s2[4]), 2)

    def test_weighted_fit_lifts_tampered_score_vs_unweighted(self):
        rng = np.random.default_rng(0)
        real = rng.normal([0.0, 0.0], 0.3, size=(40, 2))
        synth = rng.normal([3.0, 3.0], 0.3, size=(20, 2))
        tamp = rng.normal([0.4, 0.4], 0.25, size=(20, 2))
        X = np.vstack([real, synth, tamp])
        y = np.array([0] * 40 + [1] * 20 + [1] * 20)
        sid = np.array([0] * 40 + [1] * 20 + [2] * 20)
        X = np.repeat(X, 2, axis=0)
        y = np.repeat(y, 2)
        sid = np.repeat(sid, 2)

        unweighted = LogisticRegression(max_iter=2000).fit(X, y)
        weighted = fit_with_training_signal(
            X, y, sid, views_per_image=2, tampered_weight=8.0, C=1.0
        )
        probe = np.array([[0.45, 0.45]])
        p_u = unweighted.predict_proba(probe)[0, list(unweighted.classes_).index(1)]
        p_w = positive_proba(weighted, probe)[0]
        self.assertGreater(p_w, p_u)
        self.assertGreater(p_w, 0.25)
