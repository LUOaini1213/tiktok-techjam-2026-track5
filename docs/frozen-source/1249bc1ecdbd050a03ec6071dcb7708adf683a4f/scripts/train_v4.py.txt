"""Bounded train-only CV, independent calibration, development selection, pretest lock."""
from __future__ import annotations
import gc
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
from src.model import load_bundle, save_bundle, ProbabilityBlend
from src.model_v4 import FeatureHead
from src.decision import select_policy, operating_metrics
from src.decision_v4 import conditionwise_policy, original_metrics,numerical_guard
from scripts.train import calibrate
from scripts.v4_common import DATA,CACHE,REPORT,SEED,read_json,write_json,sha,preserve_v3,source_fingerprint


def auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None


def group_means(X,y,sid,groups,domains):
    unique,inv=np.unique(groups,return_inverse=True)
    if not len(X)==len(y)==len(sid)==len(groups)==len(domains): raise ValueError('Misaligned training arrays')
    for label in (y,sid):
        lo=np.full(len(unique),np.inf); hi=np.full(len(unique),-np.inf)
        np.minimum.at(lo,inv,label); np.maximum.at(hi,inv,label)
        if np.any(lo!=hi): raise ValueError('Conflicting labels within an original')
    counts=np.bincount(inv); means=np.zeros((len(unique),X.shape[1]),dtype=np.float32)
    np.add.at(means,inv,X); means/=counts[:,None]
    _,first=np.unique(groups,return_index=True)
    return means,y[first],sid[first],unique,domains[first]


def original_weights(groups,sid,domains):
    unique,first,inv,counts=np.unique(groups,return_index=True,return_inverse=True,return_counts=True)
    strata=np.array([('CF' if 'CommunityForensics' in str(d) else 'SID')+':'+str(s) for d,s in zip(domains[first],sid[first])])
    sizes={k:int((strata==k).sum()) for k in np.unique(strata)}
    w=np.array([1/sizes[k] for k in strata])[inv]/counts[inv]
    return w/w.mean()


def fit_head(kind,X,y,sid,groups,domains,C=.01):
    if kind=='mlp': X,y,sid,groups,domains=group_means(X,y,sid,groups,domains)
    weights=original_weights(groups,sid,domains)
    if kind=='hgb':
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,l2_regularization=1,
                                            early_stopping=False,random_state=SEED)
        return model.fit(X,y,sample_weight=weights)
    clf=LogisticRegression(C=C,max_iter=1000) if kind=='lr' else MLPClassifier(
        hidden_layer_sizes=(64,),alpha=.01,max_iter=80,early_stopping=False,random_state=SEED,batch_size=128)
    model=make_pipeline(StandardScaler(),clf)
    model.fit(X,y,standardscaler__sample_weight=weights,**{f'{model.steps[-1][0]}__sample_weight':weights})
    return model


def load_cache(role):
    with np.load(CACHE/f'{role}.npz') as z: return {k:z[k] for k in z.files}


def global_train():
    rows=[r for r in read_json(ROOT/'data/v3/manifest.json')['records'] if r['role']=='cached_train']
    with np.load(ROOT/'artifacts/features_train.npz') as old:
        vpi=int(old['views_per_image'][0]); idx=(np.array([r['cache_index'] for r in rows])[:,None]*vpi+np.arange(vpi)).ravel()
        X=old['X_fuse'][idx].astype(np.float32)
        y=old['y'][idx]; sid=old['sid'][idx]
    groups=np.repeat([r['group'] for r in rows],vpi); domain=np.repeat(['SID'],len(y))
    # Old CF training only. Avoid counting the explicitly duplicated SID replay twice.
    with np.load(ROOT/'artifacts/v3/train.npz') as d:
        mask=d['domain']=='OwensLab/CommunityForensics-Small'
        extra={k:d[k][mask] for k in ('X','y','sid','group','domain')}
    new=load_cache('train')
    return (np.concatenate([X,extra['X'],new['X'][:,:924]]),np.r_[y,extra['y'],new['y']],
            np.r_[sid,extra['sid'],new['sid']],np.r_[groups,extra['group'],new['group']],
            np.r_[domain,extra['domain'],new['domain']])


def bounded_cv(kind,X,y,sid,groups,domain,C):
    unique,first=np.unique(groups,return_index=True); rng=np.random.default_rng(SEED)
    chosen=[]
    for label in (0,1):
        pool=unique[y[first]==label]; chosen.extend(rng.choice(pool,min(600,len(pool)),replace=False))
    use=np.isin(groups,chosen); X,y,sid,groups,domain=(a[use] for a in (X,y,sid,groups,domain))
    folds=[]
    for tr,te in GroupKFold(3).split(X,y,groups):
        model=fit_head(kind,X[tr],y[tr],sid[tr],groups[tr],domain[tr],C)
        folds.append(auc(y[te],model.predict_proba(X[te])[:,1]))
    return folds


def summary(d,s,p):
    return dict(mean_condition_auc=float(np.mean([auc(d['y'][d['name']==n],s[d['name']==n]) for n in np.unique(d['name'])])),
                brier=float(brier_score_loss(d['y'],s)),**operating_metrics(d['y'],s,p),
                original=original_metrics(d['y'],s,d['group'],p))


def subgroup_auc(d,s,which):
    mask=d['domain']=='OwensLab/CommunityForensics-Small' if which=='external' else (
        (d['domain']=='saberzl/SID_Set') & ((d['sid']==0)|(d['sid']==2)))
    return float(np.mean([auc(d['y'][mask & (d['name']==n)],s[mask & (d['name']==n)]) for n in np.unique(d['name'])]))


def main():
    preserve_v3(); REPORT.mkdir(exist_ok=True,parents=True)
    if (REPORT/'pretest_lock.json').exists():
        from scripts.extract_v4 import validate_lock
        validate_lock(); print('Model/policy already locked; refusing test-driven refit',flush=True); return
    from scripts.validate_v4 import validate
    validate()
    fp=source_fingerprint(DATA/'manifest.json')
    cal,dev,replay,new=(load_cache(r) for r in ('cal','dev','replay','train'))
    if any(str(d['fingerprint'])!=fp for d in (cal,dev,replay,new)): raise RuntimeError('Mixed feature provenance')
    strata=cal['domain']+':'+cal['sid'].astype(str)
    unique,first=np.unique(cal['group'],return_index=True)
    prob,policy=train_test_split(unique,test_size=.5,random_state=SEED,stratify=strata[first])
    pi=np.isin(cal['group'],prob); ti=np.isin(cal['group'],policy)
    train_groups={r['group'] for r in read_json(DATA/'manifest.json')['records'] if r['role'] in ('train','replay')}
    if train_groups & (set(cal['group'])|set(dev['group'])): raise RuntimeError('Leakage')
    write_json(REPORT/'calibration_split.json',dict(probability_fit=prob.tolist(),policy=policy.tolist(),source_subclass_stratified=True))
    base=load_bundle(ROOT/'artifacts/repostguard.joblib')
    models={'v3_recal':FeatureHead(base['model'])}; trials=[]
    with threadpool_limits(limits=4):
        X,y,sid,groups,domain=global_train()
        global_training_unique=int(len(np.unique(groups)))
        global_specs=[('lr_001','lr',.01),('lr_01','lr',.1),('mlp64','mlp',.01),('hgb80','hgb',.01)]
        for name,kind,C in global_specs:
            start=time.monotonic(); folds=bounded_cv(kind,X,y,sid,groups,domain,C)
            model=fit_head(kind,X,y,sid,groups,domain,C)
            models[name]=FeatureHead(model)
            trial=dict(name=name,kind=kind,C=C,fold_auc=folds,cv_auc=float(np.mean(folds)),seconds=time.monotonic()-start,
                       training_originals=int(len(np.unique(groups))),training_rows=len(X),
                       weighting='equal SID/CF subclass strata and per-original weight; MLP fits source means')
            trials.append(trial); joblib.dump(models[name],CACHE/f'head_{name}.joblib')
            write_json(REPORT/'head_trials.json',trials); print(trial,flush=True)
        del X; gc.collect()
        local={k:np.concatenate([replay[k],new[k]]) for k in ('X','y','sid','group','domain')}
        # Local heads consume ONLY the new 824 local dimensions, enabling an explicit local-vs-global blend.
        for C in (.01,.1):
            name=f'local_lr_{C}'
            X=local['X'][:,924:]; y=local['y']; sid=local['sid']; groups=local['group']; domain=local['domain']
            folds=bounded_cv('lr',X,y,sid,groups,domain,C)
            models[name]=FeatureHead(fit_head('lr',X,y,sid,groups,domain,C),924,1748)
            trials.append(dict(name=name,fold_auc=folds,cv_auc=float(np.mean(folds)),training_originals=len(np.unique(groups)),dimensions=824))
            joblib.dump(models[name],CACHE/f'head_{name}.joblib'); write_json(REPORT/'head_trials.json',trials)
            print(trials[-1],flush=True)
    bundles={}; records={}
    def evaluate(name,model):
        calibrated=calibrate(model,cal['X'][pi],cal['y'][pi])
        # Development preview uses probability-fit groups only. These provisional
        # bounds are not final certification; final policy groups remain untouched
        # until the unique head has been selected on development below.
        cs=calibrated.predict_proba(cal['X'][pi])[:,1]
        strict=numerical_guard(select_policy(cal['y'][pi],cs,cal['group'][pi]))
        condition=conditionwise_policy(cal['y'][pi],cs,cal['group'][pi],cal['name'][pi])
        ds=calibrated.predict_proba(dev['X'])[:,1]
        inherited={k:v for k,v in base['meta'].items() if k not in (
            'metrics','selection','n_train_unique','training_data','v3_head','tampered_weight',
            'training_code_sha256','packaging_note','legacy_metadata','v2_reference_metrics')}
        bundles[name]=dict(model=calibrated,meta=dict(inherited,version='v4',feature_variant='v4_local',feature_dim=1748,
                            decision_policy=condition,strict_decision_policy=strict,seed=SEED,
                            calibration='640 disjoint probability-fit originals; 640 threshold-policy originals',
                            calibration_fit_groups=len(prob),calibration_policy_groups=len(policy),
                            n_train_unique=(12765 if name=='v3_recal' else 640 if name.startswith('local_') else global_training_unique),
                            requires_local_features='local' in name,
                            final_policy_calibrated=False,
                            calibration_scope='clean and crop_resize_jpeg only; pointwise, not simultaneous',
                            manifest_sha256=sha(DATA/'manifest.json'),feature_cache_fingerprint=fp,
                            training_code_sha256=sha(Path(__file__)),
                            training_data='SID + CommunityForensics-Small; source-normalized cached global training, explicit 640-source local replay/new-training subset',
                            weighting='equal SID/CF subclass strata and per-original weights; not a fixed 5x tampered weight',
                            v3_reference_sha256=sha(ROOT/'artifacts/repostguard.joblib'),
                            v4_head=name,test_used_for_selection=False))
        records[name]=dict(conditionwise=summary(dev,ds,condition),strict=summary(dev,ds,strict),
                           external_auc=subgroup_auc(dev,ds,'external'),tampered_auc=subgroup_auc(dev,ds,'tampered'),
                           policy=condition,strict_policy=strict)
        save_bundle(CACHE/f'{name}.joblib',**bundles[name]); print('DEV',name,records[name]['conditionwise'],flush=True)
    for name,model in models.items(): evaluate(name,model)
    best_global=max(global_specs,key=lambda r:records[r[0]]['conditionwise']['coverage'])[0]
    best_local=max([n for n in models if n.startswith('local_')],key=lambda n:records[n]['conditionwise']['coverage'])
    # Blend fixed at .5; choose its component only on dev, never the new test.
    for gname in (best_global,'v3_recal'):
        name=f'blend_{gname}_local'
        evaluate(name,FeatureHead(ProbabilityBlend(models[gname],models[best_local],.5),0,1748))
    reference=records['v3_recal']; eligible=[]
    for name,r in records.items():
        r['gate']=dict(bounds_supported=r['policy']['cal_fpr_bound_met'] and r['policy']['cal_false_clear_bound_met'],
                       external_noninferior=r['external_auc']>=reference['external_auc']-.005,
                       tampered_noninferior=r['tampered_auc']>=reference['tampered_auc']-.005,
                       lower_review=r['conditionwise']['review_rate']<reference['conditionwise']['review_rate'])
        if name!='v3_recal' and all(r['gate'].values()): eligible.append(name)
    selected=min(eligible,key=lambda n:records[n]['conditionwise']['review_rate']) if eligible else 'v3_recal'
    # Only now expose policy-group scores. No re-selection, candidate switching
    # or threshold choice based on final-calibration performance is permitted.
    for name in {selected,'v3_recal'}:
        bundle=bundles[name]; cs=bundle['model'].predict_proba(cal['X'][ti])[:,1]
        bundle['meta']['decision_policy']=conditionwise_policy(cal['y'][ti],cs,cal['group'][ti],cal['name'][ti])
        bundle['meta']['strict_decision_policy']=numerical_guard(select_policy(cal['y'][ti],cs,cal['group'][ti]))
        bundle['meta']['final_policy_calibrated']=True
        bundle['meta']['head_selected_before_final_policy']=True
        ds=bundle['model'].predict_proba(dev['X'])[:,1]
        records[name]['postselection_final_policy_diagnostic']=summary(dev,ds,bundle['meta']['decision_policy'])
        save_bundle(CACHE/f'{name}.joblib',**bundle)
    selection=dict(selected=selected,eligible=eligible,retain_v3_default=not bool(eligible),trials=records,
                   rule='dev review reduction with supported sampled-condition calibration bounds and <=.005 external/tampered mean AUC loss versus independently recalibrated V3',
                   test_used_for_selection=False,final_policy_groups_used_for_head_selection=False,
                   preview_policy_scope='Probability-fit originals reused for development preview only, not final error certification. Final policy groups first scored after unique head selection; no re-selection.',
                   protocol_sha256=sha(ROOT/'docs/v4_protocol.md'),
                   default_v3_sha256=preserve_v3(),feature_fingerprint=fp)
    write_json(REPORT/'selection.json',selection)
    save_bundle(REPORT/'frozen_pretest.joblib',**bundles[selected])
    save_bundle(REPORT/'baseline_recal.joblib',**bundles['v3_recal'])
    sources=['results/v4/frozen_pretest.joblib','results/v4/selection.json','results/v4/calibration_split.json',
             'results/v4/baseline_recal.joblib',
             'results/v4/data_manifest.json','docs/v4_protocol.md','src/features_v4.py','src/model_v4.py',
             'src/decision_v4.py','scripts/train_v4.py','scripts/extract_v4.py','scripts/prepare_v4.py']
    sources+=['scripts/train.py','src/model.py','src/score.py','scripts/v4_common.py',
              'scripts/validate_v4.py','results/v4/protocol_audit.json']
    sources+=['src/forensic_v4_fast.py','results/v4/corner_migration.json']
    write_json(REPORT/'pretest_lock.json',dict(sha256={p:sha(ROOT/p) for p in sources},selected=selected,
                                             test_used_for_selection=False,default_unchanged=True))
    print('HEAD AND POLICY LOCKED BEFORE SEALED TEST:',selected,'eligible=',eligible,flush=True)


if __name__=='__main__': main()
