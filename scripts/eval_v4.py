"""Frozen-policy matched final evaluation; no test-driven threshold or head changes."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from src.model import load_bundle
from src.decision import operating_metrics,binomial_upper
from src.decision_v4 import original_metrics
from scripts.extract_v4 import validate_lock
from scripts.v4_common import DATA,CACHE,REPORT,SEED,read_json,write_json,save_npz,sha,preserve_v3


def auc(y,s): return float(roc_auc_score(y,s))


def paired_interval(y,groups,names,s0,s1,repeats):
    unique,first=np.unique(groups,return_index=True)
    labels=y[first]; conditions=np.unique(names)
    # One index per original/condition; retain the entire transform block when resampling.
    lookup={(str(g),str(n)):i for i,(g,n) in enumerate(zip(groups,names))}
    if len(lookup)!=len(y): raise RuntimeError('Duplicate group/condition')
    idx=np.array([[lookup[(str(g),str(n))] for n in conditions] for g in unique])
    a=s0[idx]; b=s1[idx]; rng=np.random.default_rng(SEED); deltas=[]
    strata=[np.flatnonzero(labels==k) for k in (0,1)]
    for _ in range(repeats):
        sample=np.r_[*(rng.choice(x,len(x),replace=True) for x in strata)]
        ys=labels[sample]
        deltas.append(np.mean([auc(ys,b[sample,j])-auc(ys,a[sample,j]) for j in range(len(conditions))]))
    point=float(np.mean([auc(labels,b[:,j])-auc(labels,a[:,j]) for j in range(len(conditions))]))
    return dict(mean_auc_difference=point,ci95=np.quantile(deltas,[.025,.975]).tolist(),
                bootstrap_repeats=repeats,unit='stratified original-image blocks, all paired conditions retained')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--bootstrap',type=int,default=500); a=p.parse_args()
    validate_lock(); preserve_v3()
    with np.load(CACHE/'test.npz') as z: d={k:z[k] for k in z.files}
    selection=read_json(REPORT/'selection.json')
    if str(d['fingerprint'])!=selection['feature_fingerprint']: raise RuntimeError('Sealed feature fingerprint mismatch')
    if len(np.unique(d['group']))!=384 or len(d['y'])!=384*19: raise RuntimeError('Unexpected sealed size')
    v3=load_bundle(ROOT/'artifacts/repostguard.joblib')
    frozen=load_bundle(REPORT/'frozen_pretest.joblib'); recal=load_bundle(REPORT/'baseline_recal.joblib')
    scores={'v3_default':v3['model'].predict_proba(d['X'][:,:924])[:,1],
            'v3_recal':recal['model'].predict_proba(d['X'])[:,1],
            'v4_selected':frozen['model'].predict_proba(d['X'])[:,1]}
    bundles={'v3_default':v3,'v3_recal':recal,'v4_selected':frozen}
    for s in scores.values():
        if not np.isfinite(s).all(): raise RuntimeError('Nonfinite scores')
    save_npz(CACHE/'test_scores.npz',**scores,**{k:d[k] for k in ('y','sid','group','domain','name','path')})
    domains={'all':np.ones(len(d['y']),dtype=bool),
             'SID':d['domain']=='saberzl/SID_Set',
             'CF_StyleGANXL':d['domain']=='OwensLab/CommunityForensics-Small',
             'SID_tampered':(d['domain']=='saberzl/SID_Set')&((d['sid']==0)|(d['sid']==2))}
    rows=[]; summary={}; errors={}
    for name,s in scores.items():
        primary_policy='v3_frozen_strict' if name=='v3_default' else 'conditionwise'
        policies={primary_policy:bundles[name]['meta']['decision_policy']}
        if name!='v3_default': policies['strict']=bundles[name]['meta']['strict_decision_policy']
        summary[name]={}
        for domain,mask in domains.items():
            summary[name][domain]={}
            for policy_name,policy in policies.items():
                condition_rows=[]
                for condition in np.unique(d['name']):
                    use=mask&(d['name']==condition); y=d['y'][use]; ss=s[use]
                    metrics=operating_metrics(y,ss,policy); flags=int(((ss>=policy['aigc_threshold'])&(y==0)).sum())
                    clears=int(((ss<=policy['authentic_threshold'])&(y==1)).sum())
                    row=dict(model=name,domain=domain,policy=policy_name,condition=str(condition),n_originals=int(use.sum()),
                             auc=auc(y,ss),accuracy_at_05=float(accuracy_score(y,ss>=.5)),
                             **metrics,real_flag_errors=flags,n_real=int((y==0).sum()),
                             fpr_upper95=binomial_upper(flags,int((y==0).sum())),
                             fake_clear_errors=clears,n_fake=int((y==1).sum()),
                             false_clear_upper95=binomial_upper(clears,int((y==1).sum())))
                    rows.append(row); condition_rows.append(row)
                summary[name][domain][policy_name]=dict(
                    mean_auc=float(np.mean([r['auc'] for r in condition_rows])),
                    mean_accuracy_at_05=float(np.mean([r['accuracy_at_05'] for r in condition_rows])),
                    **operating_metrics(d['y'][mask],s[mask],policy),
                    original=original_metrics(d['y'][mask],s[mask],d['group'][mask],policy))
        policy=policies[primary_policy]; failure_sets={
            'false_positive_at_05':np.flatnonzero((d['y']==0)&(s>=.5)),
            'false_negative_at_05':np.flatnonzero((d['y']==1)&(s<.5)),
            'policy_false_flag':np.flatnonzero((d['y']==0)&(s>=policy['aigc_threshold'])),
            'policy_false_clear':np.flatnonzero((d['y']==1)&(s<=policy['authentic_threshold']))}
        errors[name]={}
        for kind,indices in failure_sets.items():
            chosen=[]; seen=set()
            for i in sorted(indices,key=lambda i:abs(s[i]-.5),reverse=True):
                if str(d['group'][i]) in seen: continue
                seen.add(str(d['group'][i])); chosen.append(dict(path=str(d['path'][i]),group=str(d['group'][i]),
                       condition=str(d['name'][i]),domain=str(d['domain'][i]),sid=int(d['sid'][i]),score=float(s[i])))
                if len(chosen)==5: break
            errors[name][kind]=dict(view_errors=len(indices),distinct_originals=len(np.unique(d['group'][indices])),examples=chosen)
    paired={}
    for domain,mask in domains.items():
        paired[domain]=paired_interval(d['y'][mask],d['group'][mask],d['name'][mask],scores['v3_default'][mask],scores['v4_selected'][mask],a.bootstrap)
        print(domain,paired[domain],flush=True)
    write_json(REPORT/'summary.json',dict(selected=selection['selected'],metrics=summary,paired_auc=paired,
                default_preserved=True,test_used_for_selection=False,
                calibration_scope='clean and crop_resize_jpeg only; pointwise 95%, not simultaneous',
                test_ci_caveat='192 independent real test originals cannot certify 1% FPR even at zero errors; transform rows are correlated'))
    write_json(REPORT/'error_analysis.json',errors)
    with (REPORT/'evaluation.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in scores:
        primary_policy='v3_frozen_strict' if name=='v3_default' else 'conditionwise'
        rs=[r for r in rows if r['model']==name and r['domain']=='all' and r['policy']==primary_policy]
        axes[0].plot(range(len(rs)),[r['auc'] for r in rs],'-o',label=name)
    axes[0].set_xticks(range(len(rs)),[r['condition'] for r in rs],rotation=70,fontsize=7)
    axes[0].set_ylabel('Matched test AUC'); axes[0].legend(fontsize=8)
    if selection['selected']=='v3_recal':
        axes[0].set_title('Retained classifier; ranking curves overlap')
    for name in scores:
        rs=[]; s=scores[name]
        for margin in (0,.025,.05,.1,.2,.3,.4,.45,.49):
            policy={'aigc_threshold':.5+margin,'authentic_threshold':.5-margin}
            # Predetermined margin grid. Diagnostic only; never selects a deployed threshold.
            m=operating_metrics(d['y'],s,policy); rs.append(m)
        axes[1].plot([r['coverage'] for r in rs],[r['fpr'] for r in rs],'-o',label=name)
    axes[1].set_xlabel('Automatic coverage, fixed margin grid'); axes[1].set_ylabel('Mean per-condition real FPR')
    axes[1].set_title('Post-lock diagnostic, not policy selection'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT/'comparison.png',dpi=160); plt.close(fig)
    report=['# V4: bounded CPU experiments','',f"Development selection: **{selection['selected']}**. Eligible new heads: {selection['eligible']}.",
            'The pretest head and both policies were locked before the new 384-original test was extracted.',
            'V3 remains the default throughout this research iteration. Neither historical nor new test labels train/calibrate/select the head.','',
            '| Model / policy | Mean AUC | Mean FPR | AI flag recall | False clear | Review |',
            '|---|---:|---:|---:|---:|---:|']
    if selection['selected']=='v3_recal':
        report[5:5]=['The selected fallback is exactly the independently recalibrated V3 baseline. Its duplicated rows preserve the comparison schema; they are not independent new models or trials.','']
    for name in scores:
        for pname,r in summary[name]['all'].items():
            report.append(f"| {name} / {pname} | {r['mean_auc']:.4f} | {r['fpr']:.2%} | {r['recall']:.1%} | {r['false_clear_rate']:.2%} | {r['review_rate']:.1%} |")
    report+=['','## Paired mean-AUC differences versus V3 default','',
             'Bootstrap resamples independent originals within each label, retaining all paired conditions. These are descriptive condition means, not 19 independent experiments.','']
    for domain,r in paired.items(): report.append(f"- {domain}: {r['mean_auc_difference']:+.4f}, 95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}].")
    report+=['','## Scope and limitations','',
             '- Calibration samples two conditions only. The remaining final-test transformations are out-of-calibration stress tests.',
             '- Development previews reuse probability-fit groups and are not error certification. The unique head is chosen before independent final-policy groups are scored; no candidate switch follows final calibration.',
             '- Original-count bounds assume independent/exchangeable originals within the sampled distribution; source/parent/subject dependence can violate that assumption.',
             '- Conditionwise bounds are pointwise, not a simultaneous or any-view guarantee. The strict policy counts an original wrong if any sampled view is wrong.',
             '- A calibration-supported bound is not a deployment certification. Real test sample counts cannot certify 1% even with no false flags.',
             '- CF uses identical 512x512 bicubic fit for both labels; native CLI geometry differs. FFHQ/landscape and generator/content acquisition bias can remain.',
             '- Only new CF fake model names are role-disjoint. Real-source families repeat; SID generators are not certified disjoint. StyleGAN2 was already training data, so StyleGANXL is a named-generator holdout, not an entire StyleGAN-architecture holdout.',
             '- Local pooled patch descriptors are experimental image-level features, not validated localization/explanations.',
             '- SID tampered images retain their source-level AI-positive label after transforms. Crops may remove edited regions; no mask-based post-transform relabeling is applied.',
             '- The color-jitter preset is the +20% side, not a full brightness/contrast/saturation Cartesian sweep.',
             '- Protected bitmap/near exclusion cannot prove disjoint SID/COCO parent photographs or unknown encoder pretraining.',
             '- CommunityForensics-Small is CC-BY-NC-SA-4.0: non-commercial research only. No source images or large feature caches are published.',
             '- No TechJam submission, placement, live moderation deployment, or universal robustness is claimed.','',
             '[Protocol](../../docs/v4_protocol.md) · [Selection and all failed trials](selection.json) · [Head CV](head_trials.json) · [Per-condition table](evaluation.csv) · [Error cases](error_analysis.json)','',
             '![Locked matched comparison](comparison.png)','']
    # A generated report is output data, not a hand-written source edit.
    (REPORT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    write_json(REPORT/'evaluation_lock.json',dict(pretest_lock_sha256=sha(REPORT/'pretest_lock.json'),
                evaluation_code_sha256=sha(Path(__file__)),
                report_sha256={p:sha(REPORT/p) for p in ('summary.json','evaluation.csv','error_analysis.json','comparison.png','REPORT.md')},
                test_used_for_selection=False))
    validate_lock(); preserve_v3(); print('V4 sealed evaluation completed without changing head or thresholds',flush=True)


if __name__=='__main__': main()
