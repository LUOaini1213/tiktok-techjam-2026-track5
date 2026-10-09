"""Deduplicated, grouped selection; separate probability/decision calibration; test lock."""
from __future__ import annotations
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold, train_test_split
from threadpoolctl import threadpool_limits
from src.decision import select_policy, operating_metrics
from src.model import load_bundle, save_bundle, predict_proba
from src.train_signal import fit_with_training_signal
from scripts.train import calibrate

BASE=ROOT/'artifacts/v3'
REPORT=ROOT/'results/v3'


def grouped_consistency(X,y,sid,groups):
    """Variable view counts, but the identical per-source recipe in CV and final fit."""
    extra=[]; ey=[]; es=[]; eg=[]
    order={}
    for i,g in enumerate(groups):
        order.setdefault(g,[]).append(i)
    for g,indices in order.items():
        idx=np.array(indices)
        if len(np.unique(y[idx]))!=1 or len(np.unique(sid[idx]))!=1:
            raise ValueError('Conflicting label within a source group')
        if len(idx)>1:
            extra.append(.5*(X[idx[1:]]+X[idx[0]]))
            ey.extend([y[idx[0]]]*(len(idx)-1)); es.extend([sid[idx[0]]]*(len(idx)-1)); eg.extend([g]*(len(idx)-1))
    if not extra:
        return X,y,sid,groups
    return np.concatenate([X,*extra]),np.r_[y,ey],np.r_[sid,es],np.r_[groups,eg]


def fit(X,y,sid,groups,C,weight):
    X2,y2,s2,_=grouped_consistency(X,y,sid,groups)
    return fit_with_training_signal(X2,y2,s2,views_per_image=1,C=C,tampered_weight=weight)


def _auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None


def by_transform(d,s):
    return {str(n):_auc(d['y'][d['name']==n],s[d['name']==n]) for n in np.unique(d['name'])}


def select_on_dev(base_auc,candidate_auc):
    names=list(base_auc)
    bm=float(np.mean([base_auc[n] for n in names])); cm=float(np.mean([candidate_auc[n] for n in names]))
    ok=(cm>=bm and candidate_auc['resize_0.25']>=base_auc['resize_0.25']-.005
        and candidate_auc['clean']>=base_auc['clean']-.01)
    return bool(ok),dict(mean_auc_baseline=bm,mean_auc_candidate=cm,
                         rule='candidate mean >= baseline; thumbnail regression <=.005; clean regression <=.01')


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'data/v3/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    cached=[r for r in manifest['records'] if r['role']=='cached_train']
    legacy=np.load(ROOT/'artifacts/features_train.npz')
    vpi=int(legacy['views_per_image'][0]); indices=np.array([r['cache_index'] for r in cached])
    raw_idx=(indices[:,None]*vpi+np.arange(vpi)).ravel()
    X=legacy['X_fuse'][raw_idx].astype(np.float32)
    y=legacy['y'][raw_idx]; sid=legacy['sid'][raw_idx]
    groups=np.repeat([r['group'] for r in cached],vpi)
    del legacy; gc.collect()
    extra=np.load(BASE/'train.npz')
    X=np.concatenate([X,extra['X']]); y=np.r_[y,extra['y']]; sid=np.r_[sid,extra['sid']]; groups=np.r_[groups,extra['group']]
    assert not set(groups)&{r['group'] for r in manifest['records'] if r['role'] in ('cal','dev','test')}
    # Fixed selection budget, stratified on original image identity; all augmented
    # rows for a selected image are retained. No dev, cal or test labels enter CV.
    unique,first=np.unique(groups,return_index=True)
    rng=np.random.default_rng(31009); selected=[]
    for label in (0,1):
        candidates=unique[y[first]==label]
        selected.extend(rng.choice(candidates,size=min(600,len(candidates)),replace=False))
    use=np.isin(groups,selected)
    cx,cy,cs,cg=X[use],y[use],sid[use],groups[use]
    trials=[]; start=time.monotonic()
    with threadpool_limits(limits=4):
        for C in (.01,.1,1.):
            for weight in (1.,2.,5.):
                aucs=[]
                for tr,te in GroupKFold(3).split(cx,cy,cg):
                    model=fit(cx[tr],cy[tr],cs[tr],cg[tr],C,weight)
                    aucs.append(_auc(cy[te],model.predict_proba(cx[te])[:,1]))
                trial=dict(C=C,tampered_weight=weight,cv_auc=float(np.mean(aucs)),fold_auc=aucs)
                trials.append(trial)
                print(json.dumps(trial),f'elapsed={time.monotonic()-start:.1f}s',flush=True)
        winner=max(trials,key=lambda r:r['cv_auc'])
        print('Final fit',winner,'unique sources',len(unique),'raw views',len(y),flush=True)
        candidate=fit(X,y,sid,groups,winner['C'],winner['tampered_weight'])
    del X,cx,extra; gc.collect()
    base_bundle=load_bundle(ROOT/'results/v2/repostguard.joblib')
    cal=np.load(BASE/'cal.npz'); uy,first=np.unique(cal['group'],return_index=True)
    fit_groups,policy_groups=train_test_split(uy,test_size=.5,random_state=31009,stratify=cal['y'][first])
    fit_idx=np.isin(cal['group'],fit_groups); policy_idx=np.isin(cal['group'],policy_groups)
    assert not set(fit_groups)&set(policy_groups)
    bundles={}
    for name,model in [('baseline',base_bundle['model']),('candidate',candidate)]:
        calibrated=calibrate(model,cal['X'][fit_idx],cal['y'][fit_idx])
        s=calibrated.predict_proba(cal['X'][policy_idx])[:,1]
        policy=select_policy(cal['y'][policy_idx],s,cal['group'][policy_idx])
        meta=dict(base_bundle['meta'])
        meta.update(version='v3',calibration='sigmoid, probability-fit groups disjoint from threshold-policy groups',
                    decision_policy=policy,tta_jpeg_quality=70,tta_resize_scale=.5,
                    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    training_data=('Deduplicated SID-Set + CommunityForensics-Small (LatDiff/StyleGAN2/BigGAN); no WildFake/COCO; VQDiffusion held out') if name=='candidate' else base_bundle['meta']['training_data'],
                    v3_head=name,selection=winner if name=='candidate' else 'v2 head; recalibrated',
                    n_train_unique=int(len(unique)) if name=='candidate' else 11997,
                    calibration_fit_groups=len(fit_groups),calibration_policy_groups=len(policy_groups),
                    dataset_license='CommunityForensics-Small CC-BY-NC-SA-4.0, non-commercial research; SID-Set CC-BY-4.0')
        bundles[name]=dict(model=calibrated,meta=meta)
        save_bundle(ROOT/f'artifacts/repostguard_v3_{name}.joblib',calibrated,meta)
    dev=np.load(BASE/'dev.npz')
    dev_scores={name:predict_proba(b,dev['X']) for name,b in bundles.items()}
    aucs={name:by_transform(dev,s) for name,s in dev_scores.items()}
    promote,selection=select_on_dev(aucs['baseline'],aucs['candidate'])
    selected_name='candidate' if promote else 'baseline'
    chosen=bundles[selected_name]
    save_bundle(ROOT/'artifacts/repostguard_v3.joblib',chosen['model'],chosen['meta'])
    report=dict(selected=selected_name,dev_selection=selection,dev_auc=aucs,trials=trials,
                calibration_policy={n:b['meta']['decision_policy'] for n,b in bundles.items()},
                dev_operating={n:operating_metrics(dev['y'],dev_scores[n],b['meta']['decision_policy']) for n,b in bundles.items()},
                manifest_audit=manifest['audit'],test_used_for_selection=False,
                model_sha256=hashlib.sha256((ROOT/'artifacts/repostguard_v3.joblib').read_bytes()).hexdigest(),
                manifest_sha256=chosen['meta']['manifest_sha256'],
                test_protocol='All 15 official conditions + 4 ordered chains; original-image paired bootstrap; no test-driven promotion')
    (REPORT/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)
    print('MODEL AND POLICY FROZEN; sealed-test extraction may now begin',flush=True)


if __name__=='__main__':
    main()
