"""One-shot sealed test: matched scores, fixed decision policies and paired AUC CIs."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from threadpoolctl import threadpool_limits
from src.decision import operating_metrics, binomial_upper
from src.model import load_bundle, predict_proba


def weighted_auc_samples(y, scores, negative_counts, positive_counts):
    """Exact empirical bootstrap AUC, including half credit for score ties.

    Multinomial counts are equivalent to drawing original images with replacement.
    The same counts are reused for every model, so differences remain paired.
    """
    y=np.asarray(y); scores=np.asarray(scores)
    negative=scores[y==0]; positive=scores[y==1]
    pairs=(positive[:,None]>negative[None,:]).astype(float)+.5*(positive[:,None]==negative[None,:])
    with threadpool_limits(limits=4):
        numerator=((positive_counts@pairs)*negative_counts).sum(axis=1)
    return numerator/(positive_counts.sum(axis=1)*negative_counts.sum(axis=1))


def bootstrap_aucs(y,scores,reps=1000,seed=31009):
    rng=np.random.default_rng(seed); n0=int((y==0).sum()); n1=int((y==1).sum())
    if not n0 or not n1: raise ValueError('Bootstrap AUC requires both labels')
    negative=rng.multinomial(n0,np.full(n0,1/n0),size=reps)
    positive=rng.multinomial(n1,np.full(n1,1/n1),size=reps)
    return {name:weighted_auc_samples(y,s,negative,positive) for name,s in scores.items()}


def paired_auc_ci(y,a,b,reps=1000):
    samples=bootstrap_aucs(np.asarray(y),{'a':a,'b':b},reps=reps)
    return [float(x) for x in np.percentile(samples['b']-samples['a'],[2.5,97.5])]


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    dest=ROOT/'results/v3'; selection=json.loads((dest/'selection.json').read_text(encoding='utf-8'))
    model_path=ROOT/'artifacts/repostguard_v3.joblib'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=selection['model_sha256']:
        raise RuntimeError('Model changed after the pre-test selection lock')
    if hashlib.sha256((ROOT/'data/v3/manifest.json').read_bytes()).hexdigest()!=selection['manifest_sha256']:
        raise RuntimeError('Test/data manifest changed after the selection lock')
    d=np.load(ROOT/'artifacts/v3/test.npz')
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/extract_v3_demo.py')],cwd=ROOT,check=True)
    demo=np.load(ROOT/'artifacts/v3/demo.npz')
    d={k:np.concatenate([d[k],demo[k]]) for k in ('X','y','group','name','domain','path')}
    if not np.isfinite(d['X']).all(): raise ValueError('Non-finite evaluation features')
    bundles={'v2':load_bundle(ROOT/'results/v2/repostguard.joblib'),
             'v3':load_bundle(model_path),
             'v3_candidate':load_bundle(ROOT/'artifacts/repostguard_v3_candidate.joblib')}
    scores={n:predict_proba(b,d['X']) for n,b in bundles.items()}
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Non-finite evaluation scores')
    policies={n:b['meta'].get('decision_policy',{'aigc_threshold':.5,'authentic_threshold':np.nextafter(.5,-np.inf)}) for n,b in bundles.items()}
    rows=[]; paired=[]
    for domain in ['all_sealed',*sorted(set(d['domain']))]:
        for name in sorted(set(d['name'])):
            domain_mask=(d['domain']!='WildFake reference') if domain=='all_sealed' else d['domain']==domain
            idx=(d['name']==name)&domain_mask
            y=d['y'][idx]
            if len(np.unique(y))<2: continue
            if len(np.unique(d['group'][idx]))!=len(y):
                raise ValueError('Each condition must have one row per independent original')
            bootstrap=bootstrap_aucs(y,{model:s[idx] for model,s in scores.items()})
            for model,s in scores.items():
                p=s[idx]; lo,hi=(float(v) for v in np.percentile(bootstrap[model],[2.5,97.5]))
                op=operating_metrics(y,p,policies[model])
                errors=int(((p>=policies[model]['aigc_threshold'])&(y==0)).sum())
                row=dict(domain=domain,transform=name,model=model,n=len(y),auc=float(roc_auc_score(y,p)),
                         auc_lo=lo,auc_hi=hi,accuracy_at_0_5=float(accuracy_score(y,p>=.5)),
                         brier=float(brier_score_loss(y,p)),**op,
                         test_fpr_upper95=binomial_upper(errors,int((y==0).sum())))
                rows.append(row)
            lo,hi=(float(v) for v in np.percentile(bootstrap['v3']-bootstrap['v2'],[2.5,97.5]))
            paired.append(dict(domain=domain,transform=name,n=len(y),
                               delta_auc=float(roc_auc_score(y,scores['v3'][idx])-roc_auc_score(y,scores['v2'][idx])),
                               delta_auc_lo=lo,delta_auc_hi=hi))
            print(domain,name,'v2',round(roc_auc_score(y,scores['v2'][idx]),4),
                  'v3',round(roc_auc_score(y,scores['v3'][idx]),4),flush=True)
    write_csv(dest/'evaluation.csv',rows); write_csv(dest/'paired_auc.csv',paired)
    np.savez_compressed(ROOT/'artifacts/v3/test_scores.npz',**{f'score_{n}':s for n,s in scores.items()},
                        y=d['y'],group=d['group'],name=d['name'],domain=d['domain'],path=d['path'])
    summary={}
    for domain in sorted({r['domain'] for r in rows}):
        summary[domain]={}
        for model in bundles:
            rs=[r for r in rows if r['domain']==domain and r['model']==model]
            summary[domain][model]={k:float(np.mean([r[k] for r in rs])) for k in ('auc','fpr','recall','coverage','review_rate','brier')}
            summary[domain][model]['worst_auc']=min(r['auc'] for r in rs)
    (dest/'summary.json').write_text(json.dumps(dict(selected=selection['selected'],results=summary,
        test_used_for_selection=False,confidence_note='FPR bounds are one-sided 95%; calibration bounds are distribution-specific. Small test groups cannot certify a universal 1% FPR.'),indent=2),encoding='utf-8')
    errors=[]
    high=policies['v3']['aigc_threshold']; low=policies['v3']['authentic_threshold']
    for kind,mask in [('false_positive',(d['y']==0)&(scores['v3']>=high)),
                      ('false_clear',(d['y']==1)&(scores['v3']<=low)),
                      ('false_negative_at_0_5',(d['y']==1)&(scores['v3']<.5))]:
        indices=np.flatnonzero(mask)
        # One representative per original image, rather than 19 repeated cases.
        seen=set()
        for i in indices[np.argsort(-scores['v3'][indices] if kind=='false_positive' else scores['v3'][indices])]:
            if d['group'][i] in seen: continue
            seen.add(d['group'][i])
            errors.append(dict(kind=kind,path=str(d['path'][i]),domain=str(d['domain'][i]),transform=str(d['name'][i]),
                               score=float(scores['v3'][i]),label=int(d['y'][i]),group=str(d['group'][i])))
            if len(seen)==12: break
    (dest/'error_analysis.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    manifest=json.loads((ROOT/'data/v3/manifest.json').read_text(encoding='utf-8'))
    public=dict(manifest)
    # Source ids/hashes and relative paths are sufficient; dataset images stay local.
    (dest/'data_manifest.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/report_v3.py')],cwd=ROOT,check=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
