"""Conditionwise policy with honest pointwise (not joint) original-count bounds."""
import numpy as np
from .decision import select_policy, binomial_upper, operating_metrics


def numerical_guard(policy,margin=1e-5):
    """Only shrink acceptance regions to buffer tiny batch/single-forward differences.

    Retain the unguarded order-statistic bounds conservatively; do not recompute
    a tighter data-dependent confidence bound from a reduced observed error count.
    """
    if margin<0: raise ValueError('Numerical margin must be nonnegative')
    result=dict(policy)
    result.update(aigc_threshold=policy['aigc_threshold']+margin,
                  authentic_threshold=policy['authentic_threshold']-margin,
                  numerical_score_margin=margin,
                  cal_error_counts_scope='unguarded acceptance region; counts/bounds conservative after numerical margin')
    return result


def conditionwise_policy(y,scores,groups,names,**kwargs):
    y,scores,groups,names=map(np.asarray,(y,scores,groups,names))
    if not len(y)==len(scores)==len(groups)==len(names): raise ValueError('Misaligned policy inputs')
    details={}
    for name in np.unique(names):
        mask=names==name
        if len(np.unique(groups[mask]))!=mask.sum(): raise ValueError('Duplicate original within a condition')
        details[str(name)]=numerical_guard(select_policy(y[mask],scores[mask],groups[mask],**kwargs))
    if not details: raise ValueError('No calibration conditions')
    # Conservative intersection of conditionwise acceptance regions, not pooling rows.
    result=dict(next(iter(details.values())))
    result.update(aigc_threshold=max(p['aigc_threshold'] for p in details.values()),
                  authentic_threshold=min(p['authentic_threshold'] for p in details.values()),
                  unit='original image per condition',scope='sampled calibration conditions; 95% pointwise, not simultaneous',
                  conditions=details,
                  cal_fpr_bound_met=all(p['cal_fpr_bound_met'] for p in details.values()),
                  cal_false_clear_bound_met=all(p['cal_false_clear_bound_met'] for p in details.values()))
    # Keep bounds conservative; do not inherit one arbitrary condition's error counts.
    result['cal_fpr_upper']=max(p['cal_fpr_upper'] for p in details.values())
    result['cal_false_clear_upper']=max(p['cal_false_clear_upper'] for p in details.values())
    result['cal_real_group_errors']=max(p['cal_real_group_errors'] for p in details.values())
    result['cal_fake_group_errors']=max(p['cal_fake_group_errors'] for p in details.values())
    return result


def original_metrics(y,scores,groups,policy):
    y,scores,groups=map(np.asarray,(y,scores,groups))
    real=[]; fake=[]; review=[]
    for g in np.unique(groups):
        m=groups==g
        if len(np.unique(y[m]))!=1: raise ValueError('Inconsistent group labels')
        s=scores[m]; hi=policy['aigc_threshold']; lo=policy['authentic_threshold']
        if y[m][0]==0: real.append(bool((s>=hi).any()))
        else: fake.append(bool((s<=lo).any()))
        review.append(bool(((s>lo)&(s<hi)).any()))
    return dict(real_any_view_errors=sum(real),n_real=len(real),
                real_any_view_fpr=float(np.mean(real)),real_any_view_fpr_upper=binomial_upper(sum(real),len(real)),
                fake_any_view_clear_errors=sum(fake),n_fake=len(fake),
                fake_any_view_false_clear=float(np.mean(fake)),fake_any_view_false_clear_upper=binomial_upper(sum(fake),len(fake)),
                original_any_view_review=float(np.mean(review)))
