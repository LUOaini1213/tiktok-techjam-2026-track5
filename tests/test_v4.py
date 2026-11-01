"""Cheap V4 policy/group/descriptor contract tests; no model or data download."""
import numpy as np
import pytest
import torch
from sklearn.dummy import DummyClassifier
from src.decision import select_policy
from src.decision_v4 import conditionwise_policy,original_metrics
from src.features_v4 import pool_patch_tokens,corner_summary,LOCAL_DIM,V4_DIM
from src.model_v4 import FeatureHead
from scripts.train_v4 import group_means,original_weights


def policy_data():
    # 320 independent sources/class, two paired conditions; zero errors supports 1% pointwise.
    y=np.repeat(np.r_[np.zeros(320,dtype=int),np.ones(320,dtype=int)],2)
    groups=np.repeat(np.arange(640),2); names=np.tile(['clean','chain'],640)
    s=np.where(y==0,.1,.9)+np.tile([-.01,.01],640)
    return y,s,groups,names


def test_conditionwise_counts_are_originals_not_correlated_rows():
    y,s,g,n=policy_data(); p=conditionwise_policy(y,s,g,n)
    assert p['n_real_groups']==320 and p['n_fake_groups']==320
    assert p['cal_fpr_bound_met'] and 'not simultaneous' in p['scope']
    assert all(r['n_real_groups']==320 for r in p['conditions'].values())
    assert original_metrics(y,s,g,p)['n_real']==320


def test_tiny_calibration_does_not_certify_one_percent():
    y,s,g,n=policy_data(); keep=(g<10)|((g>=320)&(g<330))
    assert not conditionwise_policy(y[keep],s[keep],g[keep],n[keep])['cal_fpr_bound_met']


def test_duplicate_condition_original_rejected():
    y,s,g,n=policy_data()
    with pytest.raises(ValueError): conditionwise_policy(y,s,g,np.repeat('clean',len(y)))


def test_strict_and_conditionwise_are_distinct_scopes():
    y,s,g,n=policy_data(); a=select_policy(y,s,g); b=conditionwise_policy(y,s,g,n)
    assert a['unit']!=b['unit']
    assert b['aigc_threshold']>=max(r['aigc_threshold'] for r in b['conditions'].values())
    from src.decision_v4 import numerical_guard
    guarded=numerical_guard(a)
    assert guarded['aigc_threshold']>a['aigc_threshold'] and guarded['authentic_threshold']<a['authentic_threshold']
    assert guarded['cal_fpr_upper']==a['cal_fpr_upper']


def test_patch_pool_is_finite_and_excludes_cls_by_caller():
    t=torch.ones((2,1,384)); p=pool_patch_tokens(t)
    assert p.shape==(2,768) and torch.isfinite(p).all() and torch.count_nonzero(p)==0
    with pytest.raises(ValueError): pool_patch_tokens(torch.ones(2,0,384))


def test_native_corner_summary_handles_tiny_constant_image():
    from PIL import Image
    s=corner_summary(Image.new('RGB',(3,5),(80,80,80)))
    assert s.shape==(56,) and np.isfinite(s).all() and np.allclose(s[:28],0)
    assert LOCAL_DIM==824 and V4_DIM==1748


def test_memory_efficient_center_is_exactly_v3_pixels():
    from PIL import Image
    from src.features_v4 import center_forensic
    from src.features import forensic_vector
    rng=np.random.default_rng(4)
    for h,w in ((301,503),(81,90),(300,70)):
        im=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
        np.testing.assert_array_equal(center_forensic(im),forensic_vector(im))
        from src.forensic_v4_fast import forensic_vector_fast
        np.testing.assert_array_equal(forensic_vector_fast(im),forensic_vector(im))


def test_v4_view_grid_and_streaming_budget():
    from PIL import Image
    from scripts.extract_v4 import view_set
    im=Image.new('RGB',(32,32))
    assert len(list(view_set(im,'test')))==19
    assert len(list(view_set(im,'dev')))==6
    assert [n for n,_ in view_set(im,'cal')]==['clean','crop_resize_jpeg']


def test_adapter_slices_and_rejects_wrong_width():
    m=DummyClassifier().fit(np.zeros((4,28)),[0,1,0,1]); h=FeatureHead(m,896,924)
    assert h.predict_proba(np.zeros((2,1748))).shape==(2,2)
    with pytest.raises(ValueError): h.predict_proba(np.zeros((2,924)))
    from scripts.train import calibrate
    X=np.zeros((20,1748)); y=np.tile([0,1],10)
    calibrated=calibrate(h,X,y)
    assert calibrated.predict_proba(X).shape==(20,2)
    with pytest.raises(RuntimeError): h.fit(X,y)
    from src.model_v4 import GlobalPadAdapter
    packaged=GlobalPadAdapter(h)
    np.testing.assert_array_equal(packaged.predict_proba(X[:,:924]),h.predict_proba(X))


def test_group_mean_and_weights_are_view_count_invariant():
    X=np.array([[1.,2.],[3.,4.],[5.,6.]]); y=np.array([0,0,1]); sid=y; g=np.array(['a','a','b']); d=np.array(['SID']*3)
    means,yy,ss,gg,dd=group_means(X,y,sid,g,d)
    assert np.allclose(means,[[2,3],[5,6]])
    w=original_weights(g,sid,d)
    assert w[g=='a'].sum()==pytest.approx(w[g=='b'].sum())
    with pytest.raises(ValueError): group_means(X,np.array([0,1,1]),sid,g,d)


def test_monotone_calibration_preserves_rank_auc():
    from sklearn.metrics import roc_auc_score
    y=np.array([0,1,0,1,1,0]); s=np.array([.1,.4,.3,.8,.7,.6])
    sigmoid=1/(1+np.exp(-3*(s-.5)))
    assert roc_auc_score(y,s)==roc_auc_score(y,sigmoid)


def test_v4_cli_passes_artifact_variant(monkeypatch,tmp_path):
    import infer
    from PIL import Image
    Image.new('RGB',(4,4)).save(tmp_path/'sample.png')
    bundle={'model':object(),'meta':dict(feature_variant='v4_local',feature_dim=1748,use_forensic=True)}
    monkeypatch.setattr(infer,'load_bundle',lambda _:bundle)
    def embed(image,**kw):
        assert kw['variant']=='v4_local'; return np.zeros(1748)
    monkeypatch.setattr(infer,'embed_for_score',embed)
    monkeypatch.setattr(infer,'score_from_embed',lambda b,x:.42)
    rows=infer.score_directory(tmp_path,tmp_path/'out.json')
    assert len(rows)==1 and rows[0]['pred']==.42


def test_atomic_cache_allows_path_metadata(tmp_path):
    from scripts.v4_common import save_npz
    destination=tmp_path/'cache.npz'
    save_npz(destination,X=np.zeros((2,1748)),path=np.array(['a.png','b.png']))
    with np.load(destination) as d: assert d['path'].tolist()==['a.png','b.png']


def test_retained_classifier_release_metadata_is_not_candidate_training():
    from scripts.verify_v4 import release_meta
    policy={'aigc_threshold':.9,'authentic_threshold':.1}
    original=dict(v4_head='v3_recal',training_code_sha256='new-selection-code',
                  training_data='new candidate sources',decision_policy=policy)
    base=dict(training_code_sha256='v3-fit-code',training_data='v3 sources',
              tampered_weight=1.,seed=31009,manifest_sha256='v3-manifest',selection={'C':.01})
    corrected=release_meta(original,base)
    assert corrected['training_data']=='v3 sources'
    assert corrected['calibration_selection_code_sha256']=='new-selection-code'
    assert corrected['training_code_sha256']=='v3-fit-code'
    assert corrected['decision_policy'] is policy
    assert original['training_data']=='new candidate sources'


def test_demo_explicit_missing_weights_do_not_fall_back(monkeypatch,tmp_path):
    import app
    calls=[]
    def missing(weights=None):
        calls.append(weights)
        raise FileNotFoundError(str(weights))
    monkeypatch.setattr(app,'load_weights',missing)
    explicit=tmp_path/'missing-experimental.joblib'
    with pytest.raises(FileNotFoundError): app.build_demo(explicit)
    assert calls==[explicit]


def test_release_fixtures_ignore_gitkeep_and_support_jpeg(tmp_path):
    from PIL import Image
    from scripts.verify_v4 import sample_fixtures
    (tmp_path/'.gitkeep').touch()
    (tmp_path/'README.md').write_text('Not a test image',encoding='utf-8')
    for name in ('a.jpg','b.jpg'): Image.new('RGB',(4,4)).save(tmp_path/name)
    fixtures=sample_fixtures(tmp_path)
    assert len(fixtures)==2 and all(im.size==(4,4) for im in fixtures)
