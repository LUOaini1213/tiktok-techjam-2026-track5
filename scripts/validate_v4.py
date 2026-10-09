"""Reproducible decoded-source audit, independent of model predictions."""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.v4_common import DATA,CACHE,REPORT,read_json,write_json,sha
from scripts.audit_v3 import identity
from src.io_utils import open_image


def validate():
    m=read_json(DATA/'manifest.json'); rows=m['records']
    if not m['complete'] or m['counts']!=dict(train=256,dev=256,cal=1280,test=384,replay=384):
        raise RuntimeError('Incomplete declared quotas')
    groups={}
    for r in rows:
        prior=groups.setdefault(r['group'],(r['role'],r['label']))
        if prior!=(r['role'],r['label']): raise RuntimeError('Original crosses role/label')
        if r['role']!='test' and r['model'] in ('StyleGANXL','VQDiffusion'): raise RuntimeError('Held-out generator leakage')
        if 'COCO' in r['model'] or 'WildFake' in r['dataset']: raise RuntimeError('Protected reference source deliberately sampled')
    old=read_json(CACHE/'decoded_audit.json') if (CACHE/'decoded_audit.json').exists() else {}
    records={}
    for i,r in enumerate(rows):
        path=ROOT/r['path']; stamp=f'{path.stat().st_size}:{path.stat().st_mtime_ns}'
        cached=old.get(r['path'])
        digest=cached['sha256'] if cached and cached['stamp']==stamp else identity(open_image(path))[0]
        if digest!=r['group']: raise RuntimeError(f'Stored pixels differ from frozen source identity: {r["path"]}')
        records[r['path']]=dict(stamp=stamp,sha256=digest)
        if (i+1)%512==0: print(f'Decoded V4 identity audit {i+1}/{len(rows)}',flush=True)
    write_json(CACHE/'decoded_audit.json',records)
    result=dict(manifest_sha256=sha(DATA/'manifest.json'),decoded_sources_checked=len(rows),
                stored_pixel_identity_mismatches=0,protected_identity_inventory=m['protected_old_originals'],
                excluded_exact_or_near=m['excluded'],all_new_groups_and_roles_disjoint=True,
                heldout_generator='StyleGANXL',official_reference_used_for_training=False,
                source_family_disjoint_scope='New CF fake model names only. FFHQ/LandscapesHQ real families repeat across roles; SID generator families are not certified disjoint. StyleGAN2 was already training data: this is not an entire StyleGAN-architecture holdout.',
                parent_overlap_certified=False)
    write_json(REPORT/'protocol_audit.json',result)
    return result


if __name__=='__main__': print(validate())
