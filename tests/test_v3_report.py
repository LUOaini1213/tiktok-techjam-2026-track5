"""Report/chart generation stays reproducible without downloading any dataset."""
import csv
import json
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts import report_v3
from scripts.eval_v3 import weighted_auc_samples,paired_auc_ci
from src.decision import binomial_upper,select_policy


def test_matched_report_and_figure(tmp_path,monkeypatch):
    monkeypatch.setattr(report_v3,'DEST',tmp_path)
    models=('v2','v3'); conditions=('clean','resize_0.25','jpeg_30','noise_0.05','crop_resize_jpeg','double_repost')
    rows=[dict(domain=d,model=m,transform=t,n=400,auc=.8,auc_lo=.7,auc_hi=.9,fpr=.01,recall=.4,review_rate=.5)
          for d in report_v3.DOMAINS for m in models for t in conditions]
    with (tmp_path/'evaluation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={d:{m:dict(auc=.8,worst_auc=.7,fpr=.01,recall=.4,review_rate=.5) for m in models} for d in report_v3.DOMAINS}
    (tmp_path/'summary.json').write_text(json.dumps(dict(results=summary)),encoding='utf-8')
    (tmp_path/'selection.json').write_text(json.dumps(dict(selected='baseline',model_sha256='fixture-only')),encoding='utf-8')
    report_v3.main()
    assert (tmp_path/'comparison.png').stat().st_size>1000
    text=(tmp_path/'REPORT.md').read_text(encoding='utf-8')
    assert 'fixture-only' in text and 'abstentions' in text and '512x512' in text


def test_invalid_statistical_targets_are_not_silently_certified():
    with pytest.raises(ValueError,match='Confidence'):
        binomial_upper(0,400,confidence=1)
    with pytest.raises(ValueError,match='targets'):
        select_policy([0,1],[.1,.9],[0,1],target_fpr=-1)


def test_vectorized_bootstrap_equals_roc_auc_on_the_same_resampled_originals():
    y=np.array([0,0,0,1,1]); score=np.array([.1,.5,.8,.5,.9])
    negative=np.array([[1,1,1],[0,1,2],[3,0,0]])
    positive=np.array([[1,1],[2,0],[0,2]])
    got=weighted_auc_samples(y,score,negative,positive)
    expected=[roc_auc_score(y,score,sample_weight=np.r_[n,p]) for n,p in zip(negative,positive)]
    np.testing.assert_allclose(got,expected,atol=1e-12)
    assert paired_auc_ci(y,score,score,reps=30)==[0.,0.]
