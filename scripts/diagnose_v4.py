"""Training-only branch ablations and reused-calibration diagnostic risk curves."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.model import load_bundle
from src.decision import select_policy, operating_metrics
from src.decision_v4 import original_metrics
from scripts.v4_common import REPORT,SEED,read_json,write_json,preserve_v3


def training_sample(n_per_class=600):
    rows=[r for r in read_json(ROOT/'data/v3/manifest.json')['records'] if r['role']=='cached_train']
    rng=np.random.default_rng(SEED); chosen=[]
    for label in (0,1):
        idx=[i for i,r in enumerate(rows) if r['label']==label]
        chosen.extend(rng.choice(idx,min(n_per_class,len(idx)),replace=False))
    rows=[rows[i] for i in chosen]
    with np.load(ROOT/'artifacts/features_train.npz') as old:
        vpi=int(old['views_per_image'][0]); indices=(np.array([r['cache_index'] for r in rows])[:,None]*vpi+np.arange(vpi)).ravel()
        X=old['X_fuse'][indices].astype(np.float32)
    return X,np.repeat([r['label'] for r in rows],vpi),np.repeat([r['group'] for r in rows],vpi)


def main():
    preserve_v3(); out=REPORT/'diagnosis'; out.mkdir(parents=True,exist_ok=True)
    X,y,g=training_sample()
    branches={'clip':np.arange(512),'dino':np.arange(512,896),'forensic':np.arange(896,924),
              'clip_dino':np.arange(896),'clip_forensic':np.r_[0:512,896:924],
              'dino_forensic':np.arange(512,924),'full':np.arange(924)}
    trials=[]
    with threadpool_limits(limits=4):
        for name,columns in branches.items():
            scores=[]
            for tr,te in GroupKFold(3).split(X,y,g):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000))
                model.fit(X[tr][:,columns],y[tr]); scores.append(float(roc_auc_score(y[te],model.predict_proba(X[te][:,columns])[:,1])))
            row=dict(branch=name,dimensions=len(columns),fold_auc=scores,mean_auc=float(np.mean(scores)))
            trials.append(row); write_json(out/'ablations.json',dict(unit='1200 unique SID training originals; grouped 3-fold; all six views kept together',trials=trials))
            print(row,flush=True)
    del X
    bundle=load_bundle(ROOT/'artifacts/repostguard.joblib'); curves=[]
    for role in ('dev','cal'):
        with np.load(ROOT/f'artifacts/v3/{role}.npz') as d:
            y=d['y']; scores=bundle['model'].predict_proba(d['X'])[:,1]; groups=d['group']
            for target in (.001,.005,.01,.02,.05,.1):
                policy=select_policy(y,scores,groups,target_fpr=target)
                curves.append(dict(cache=role,target_fpr=target,**operating_metrics(y,scores,policy),
                                   **original_metrics(y,scores,groups,policy),
                                   fpr_bound_supported=policy['cal_fpr_bound_met'],
                                   fpr_upper=policy['cal_fpr_upper']))
    write_json(out/'risk_coverage.json',dict(caveat='Same-label diagnostic only. V3 calibration has been reused; this is not independent validation or a deployment guarantee.',rows=curves))
    with (out/'risk_coverage.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].barh([r['branch'] for r in trials],[r['mean_auc'] for r in trials]); axes[0].set_xlim(.5,1)
    axes[0].set_xlabel('Grouped training CV AUC (not test)')
    for role in ('dev','cal'):
        rows=[r for r in curves if r['cache']==role]
        axes[1].plot([r['coverage'] for r in rows],[r['real_any_view_fpr_upper'] for r in rows],'-o',label=role)
    axes[1].set_xlabel('Automatic coverage (view mean)'); axes[1].set_ylabel('Real any-view FPR upper bound')
    axes[1].set_title('Reused-label diagnostic only'); axes[1].legend()
    fig.tight_layout(); fig.savefig(out/'diagnosis.png',dpi=160); plt.close(fig)
    preserve_v3()


if __name__=='__main__': main()
