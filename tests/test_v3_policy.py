import random
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from src.decision import select_policy, verdict, operating_metrics
from src.repost import train_views, apply_chain
from scripts.train import cv_auc


def test_policy_counts_originals_not_augmented_rows():
    y = np.repeat([0] * 400 + [1] * 400, 3)
    groups = np.repeat(np.arange(800), 3)
    scores = np.repeat(np.r_[np.linspace(.01, .4, 400), np.linspace(.6, .99, 400)], 3)
    p = select_policy(y, scores, groups)
    assert p['n_real_groups'] == 400
    assert p['cal_fpr_upper'] <= .01
    assert p['cal_false_clear_upper'] <= .05
    assert verdict(.5, p) == 'Needs review'
    assert operating_metrics(y, scores, p)['fpr'] <= .01


def test_small_calibration_set_does_not_claim_one_percent_guarantee():
    p = select_policy([0, 0, 1, 1], [.1, .2, .8, .9], range(4))
    assert not p['cal_fpr_bound_met']
    assert p['cal_fpr_upper'] > .01
    assert verdict(float('nan'), p) == 'Needs review'


def test_group_label_conflicts_are_rejected():
    with pytest.raises(ValueError):
        select_policy([0, 1], [.1, .9], [0, 0])


def test_cv_uses_final_consistency_recipe_without_crossing_groups():
    X = np.arange(36, dtype=float).reshape(12, 3)
    y = np.repeat([0, 1, 0, 1, 0, 1], 2)
    groups = np.repeat(np.arange(6), 2)
    seen = []
    def fit(a, b, c, **kwargs):
        seen.append(kwargs)
        return object()
    with patch('scripts.train.fit_with_training_signal', side_effect=fit), patch('scripts.train.positive_proba', side_effect=lambda m, a: np.linspace(0, 1, len(a))):
        cv_auc(X, y, y, groups, 2, 1, 3, tampered_weight=2)
    assert seen and all(k['views_per_image'] == 2 and k['tampered_weight'] == 2 for k in seen)


def test_chains_and_training_views_are_reproducible():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (80, 96, 3), dtype=np.uint8))
    a, b = train_views(image, 10), train_views(image, 10)
    assert len(a) == 7
    for (name, av), (name2, bv) in zip(a, b):
        assert name == name2 and av.size == image.size
        np.testing.assert_array_equal(av, bv)
    assert not np.array_equal(image, apply_chain(image, 'double_repost'))
