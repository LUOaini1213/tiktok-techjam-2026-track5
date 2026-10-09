"""Generate a compact report and scientific figure from locked matched-test scores."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/v3'
DOMAINS=['SID-Set fresh','OwensLab/CommunityForensics-Small','WildFake reference']
LABELS=['Fresh SID (19 conditions)','Unseen VQDiffusion (19)','Official reference (6)']


def main():
    rows=list(csv.DictReader((DEST/'evaluation.csv').open(encoding='utf-8',newline='')))
    summary=json.loads((DEST/'summary.json').read_text(encoding='utf-8'))
    selected=json.loads((DEST/'selection.json').read_text(encoding='utf-8'))
    lookup={(r['domain'],r['transform'],r['model']):r for r in rows}
    lines=['# v3 matched evaluation','',
           'Post-deadline research, never a competition submission. Model and thresholds were locked before the new test was scored.',
           '',f"Selected on development only: `{selected['selected']}`. Model SHA256: `{selected['model_sha256']}`.",
           '', 'The exact [pre-test artifact](frozen_pretest.joblib) is archived. Packaging only corrects inherited documentation fields; [preflight](preflight.json) verifies identical estimator, inference policy and sealed predictions, and records both artifact checksums.',
           '', '## Domain summaries','',
           'Means below are descriptive averages over conditions, not independent trials. v2/v3 use exactly the same originals within each row. Domains have different transform counts. Community Forensics inputs are uniformly fit to 512x512 before transforms; those scores do not describe native source files.',
           '', '| Domain | Originals/condition | Model | Mean AUC | Worst AUC | FPR | Recall | Review rate |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for domain in DOMAINS:
        for model in ('v2','v3'):
            r=summary['results'][domain][model]; n=lookup[domain,'clean',model]['n']
            lines.append(f"| {domain} | {n} | {model} | {r['auc']:.4f} | {r['worst_auc']:.4f} | {r['fpr']:.1%} | {r['recall']:.1%} | {r['review_rate']:.1%} |")
    lines+=['','v2 uses its original binary 0.5 threshold. v3 uses a calibrated three-way policy. Lower FPR can cost recall and coverage; review cases are abstentions, not correct predictions.',
            '', '![Matched AUC and review trade-off](comparison.png)',
            '', '## Fixed 0.5 threshold diagnostics','',
            'These diagnostics separate changes to the model score from changes to the review policy. This is not the default v3 operating policy.',
            '', '| Domain | Model | Mean accuracy | Mean FPR | Mean AI recall |',
            '|---|---|---:|---:|---:|']
    for domain in DOMAINS:
        for model in ('v2','v3'):
            r=summary['results'][domain][model]
            lines.append(f"| {domain} | {model} | {r['accuracy_at_0_5']:.1%} | {r['fpr_at_0_5']:.1%} | {r['recall_at_0_5']:.1%} |")
    lines+=[
            '', '## Selected conditions','',
            '| Domain | Condition | v2 AUC | v3 AUC [95% CI] | v3 FPR | v3 recall | v3 review |',
            '|---|---|---:|---:|---:|---:|---:|']
    for domain in DOMAINS:
        for condition in ('clean','resize_0.25','jpeg_30','noise_0.05','crop_resize_jpeg','double_repost'):
            a=lookup[domain,condition,'v2']; b=lookup[domain,condition,'v3']
            lines.append(f"| {domain} | {condition} | {float(a['auc']):.4f} | {float(b['auc']):.4f} [{float(b['auc_lo']):.4f}, {float(b['auc_hi']):.4f}] | {float(b['fpr']):.1%} | {float(b['recall']):.1%} | {float(b['review_rate']):.1%} |")
    lines+=['', 'Full conditions: [evaluation.csv](evaluation.csv). Matched paired AUC-difference intervals: [paired_auc.csv](paired_auc.csv). Source roles and hashes: [data_manifest.json](data_manifest.json).',
            '', '## Full synthesis versus local edits','',
            'The historical SID development guard samples by binary label, not by tampered subtype; its easy synthetic rows must not stand in for local-edit performance. New-test subtype diagnostics are reported separately, never used to change the locked model.',
            '', '| New SID task | Model | Mean AUC | Worst AUC | AI recall at shipped policy |',
            '|---|---|---:|---:|---:|']
    for domain in ('SID-Set fresh (fully synthetic)','SID-Set fresh (tampered)'):
        if domain not in summary['results']: continue
        for model in ('v2','v3'):
            r=summary['results'][domain][model]
            lines.append(f"| {domain} | {model} | {r['auc']:.4f} | {r['worst_auc']:.4f} | {r['recall']:.1%} |")
    lines+=[
            '', '## Error analysis','',
            'Representative mistakes are listed in [error_analysis.json](error_analysis.json), with one case per original per mistake category. False-clear/false-flag errors use the shipped review policy; false negatives at 0.5 are separately labelled diagnostics.',
            '', 'Low confidence does not prove authenticity. Small real-image test groups cannot certify a universal 1% FPR. The held-out generator and real-source choices are narrow; acquisition/content bias and unseen pretraining overlap can remain. SID uses COCO-derived captions/edits: no protected reference is deliberately sampled for training, but shared parent photos cannot be ruled out. Bitmap deduplication is not parent-photo disjointness.',
            '', '## Licensing and provenance','',
            'CommunityForensics-Small is CC-BY-NC-SA-4.0. The v3 artifact is for non-commercial research; do not infer commercial usage rights from this public repository. Dataset images are not redistributed. SID-Set is CC-BY-4.0.',
            '', 'The historical v2 cache had 4,003 repeated training files and 19 train/validation overlaps. Those historical metrics remain diagnostics only. v3 excludes training matches and uses fresh, sealed tests. See [protocol](../../docs/v3_protocol.md).',
            '']
    (DEST/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(DOMAINS)); width=.34
    for i,model in enumerate(('v2','v3')):
        auc=[summary['results'][d][model]['auc'] for d in DOMAINS]
        axes[0].bar(x+(i-.5)*width,auc,width,label=model,color=('#94a3b8','#2563eb')[i])
    axes[0].set_ylim(0,1); axes[0].axhline(.5,color='#64748b',linestyle='--',linewidth=.8)
    axes[0].set_ylabel('Mean ROC AUC'); axes[0].set_title('Matched ranking performance (chance = 0.5)'); axes[0].legend()
    recall=np.array([summary['results'][d]['v3']['recall'] for d in DOMAINS])
    review=np.array([summary['results'][d]['v3']['review_rate'] for d in DOMAINS])
    axes[1].bar(x-width/2,100*recall,width,label='AI recall',color='#2563eb')
    axes[1].bar(x+width/2,100*review,width,label='Review rate',color='#f59e0b')
    axes[1].set_ylim(0,110); axes[1].set_ylabel('Percent'); axes[1].set_title('v3: false-flag reduction has a cost')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.26),ncol=2,frameon=False)
    for i,d in enumerate(DOMAINS):
        axes[1].text(i,101,f"FPR {summary['results'][d]['v3']['fpr']:.1%}",ha='center',fontsize=9)
    for ax in axes:
        ax.set_xticks(x,LABELS,fontsize=8); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('RepostGuard v3 — locked model, new tests; no test-driven tuning',fontsize=12)
    fig.savefig(DEST/'comparison.png',dpi=160); plt.close(fig)
    print('Wrote results/v3/REPORT.md and comparison.png')


if __name__=='__main__': main()
