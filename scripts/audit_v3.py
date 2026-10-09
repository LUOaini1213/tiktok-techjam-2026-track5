"""Content identities and conservative near-duplicate exclusion before the split is frozen."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from src.io_utils import open_image


def identity(image):
    rgb=np.asarray(image.convert('RGB'))
    sha=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    thumb=np.asarray(image.convert('L').resize((32,32)),dtype=np.float32)
    low=cv2.dct(thumb)[:8,:8].ravel()
    ph=int(''.join('1' if x else '0' for x in low>np.median(low[1:])),2)
    small=np.asarray(image.convert('L').resize((9,8)))
    dh=int(''.join('1' if x else '0' for x in (small[:,1:]>small[:,:-1]).ravel()),2)
    return sha,ph,dh


def audit_manifest(root,manifest):
    if manifest['audit'].get('content_audit_complete'):
        return manifest
    print('Content audit: decoded-pixel identity + pHash/dHash checks against every training source',flush=True)
    local=root/'artifacts/v3/content_identities.json'; local.parent.mkdir(parents=True,exist_ok=True)
    cache=json.loads(local.read_text(encoding='utf-8')) if local.exists() else {}
    rows=manifest['records']; path_ids={}; train_ids={}; aliases={}; kept=[]; pixel_duplicates=0
    for i,r in enumerate(rows):
        if r['role'] not in ('cached_train','train'): continue
        p=r['path']; stamp=str((root/p).stat().st_size)+':'+str((root/p).stat().st_mtime_ns)
        if p in cache and cache[p]['stamp']==stamp:
            sha,ph,dh=cache[p]['identity']
        else:
            sha,ph,dh=identity(open_image(root/p))
            cache[p]=dict(stamp=stamp,identity=[sha,ph,dh])
        aliases[r['group']]=sha; path_ids[p]=sha
        if r['role']=='cached_train' and sha in train_ids:
            pixel_duplicates+=1; continue
        train_ids[sha]=(ph,dh)
        r=dict(r,group=sha); kept.append(r)
        if (i+1)%1000==0:
            print(f'  training identities {i+1}',flush=True)
            local.write_text(json.dumps(cache),encoding='utf-8')
    local.write_text(json.dumps(cache),encoding='utf-8')
    excluded=[]
    for r in rows:
        if r['role'] in ('cached_train','train'): continue
        sha,ph,dh=identity(open_image(root/r['path']))
        near=sha in train_ids or any((ph^p).bit_count()<=4 and (dh^d).bit_count()<=2 for p,d in train_ids.values())
        if near:
            excluded.append(dict(path=r['path'],role=r['role'],group=r['group'],reason='training exact/near-duplicate'))
        else:
            kept.append(dict(r,group=sha))
    manifest['records']=kept
    manifest['audit'].update(content_audit_complete=True,decoded_training_duplicates_removed=pixel_duplicates,
                             train_to_other_near_duplicates_excluded=excluded,
                             counts_by_role=dict(Counter(r['role'] for r in kept)))
    # All identity aliases of one original stay on one side of the split.
    train_groups={r['group'] for r in kept if r['role'] in ('cached_train','train')}
    assert not train_groups&{r['group'] for r in kept if r['role'] not in ('cached_train','train')}
    print('Content audit complete:',pixel_duplicates,'extra decoded duplicates;',len(excluded),'cross-split near matches excluded',flush=True)
    return manifest
