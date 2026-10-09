import random
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from src.decision import select_policy, verdict, operating_metrics
from src.repost import train_views, apply_chain
from scripts.train import cv_auc
from scripts.train_v3 import grouped_consistency
from src.model import ProbabilityBlend
from sklearn.dummy import DummyClassifier
import joblib
from infer import prediction_row
from src.score import score_from_embed


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


def test_variable_view_consistency_stays_with_original_and_rejects_label_conflicts():
    X=np.array([[0.],[2.],[10.],[14.],[18.]])
    y=np.array([0,0,1,1,1]); groups=np.array(['a','a','b','b','b'])
    xx,yy,ss,gg=grouped_consistency(X,y,y,groups)
    np.testing.assert_array_equal(xx[-3:,0],[1,12,14])
    np.testing.assert_array_equal(gg[-3:],['a','b','b'])
    with pytest.raises(ValueError):
        grouped_consistency(X,np.array([0,1,1,1,1]),y,groups)


def test_blended_head_is_portable_and_preserves_probability_contract(tmp_path):
    X=np.zeros((2,3)); y=np.array([0,1])
    a=DummyClassifier(strategy='constant',constant=0).fit(X,y)
    b=DummyClassifier(strategy='constant',constant=1).fit(X,y)
    blend=ProbabilityBlend(a,b,.25)
    path=tmp_path/'blend.joblib'; joblib.dump(blend,path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X),[[.75,.25],[.75,.25]])


def test_nonfinite_values_cannot_silently_become_confident_authentic_scores():
    assert prediction_row('bad.jpg',float('nan'))['pred']==.5
    with pytest.raises(ValueError,match='Non-finite'):
        score_from_embed({},np.array([np.nan,1.]))
