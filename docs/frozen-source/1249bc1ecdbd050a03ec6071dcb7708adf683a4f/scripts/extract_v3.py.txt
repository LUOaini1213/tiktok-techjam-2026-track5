"""Checkpoint each source image; never extract the sealed test during model selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageOps
from src.augment import EVAL_PRESETS, apply_named
from src.config import load_config
from src.io_utils import open_image
from src.repost import CHAINS, apply_chain, train_views
from src.score import embed_for_score
from scripts.audit_v3 import audit_manifest


def view_set(image, role, group):
    if role == 'train':
        return train_views(image, f'31009:{group}')
    wanted = {'clean','jpeg_30','blur_1.0','resize_0.25','noise_0.05'}
    if role == 'test':
        wanted = {n for n,_,_ in EVAL_PRESETS}
    elif role == 'dev':
        wanted.add('jitter_0.20')
    views = [(n,apply_named(image,op,p)) for n,op,p in EVAL_PRESETS if n in wanted]
    names = list(CHAINS) if role in ('dev','test') else ['crop_resize_jpeg']
    return views + [(n,apply_chain(image,n)) for n in names]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--roles',default='train,cal,dev'); p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads)
    cfg=load_config(); base=ROOT/'artifacts/v3'; base.mkdir(exist_ok=True,parents=True)
    manifest_path=ROOT/'data/v3/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise SystemExit('Dataset manifest is not complete')
    if not manifest['audit'].get('content_audit_complete'):
        manifest=audit_manifest(ROOT,manifest)
        tmp=manifest_path.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); os.replace(tmp,manifest_path)
    fp=hashlib.sha256(manifest_path.read_bytes()+b''.join((ROOT/f).read_bytes() for f in ('src/features.py','src/repost.py','src/augment.py','scripts/extract_v3.py'))).hexdigest()
    for role in args.roles.split(','):
        records=[r for r in manifest['records'] if r['role']==role]
        chunks=[]; ys=[]; sids=[]; groups=[]; domains=[]; names=[]; paths=[]
        start=time.monotonic()
        for i,r in enumerate(records):
            target=base/'images'/f'{role}_{r["group"]}.npz'; target.parent.mkdir(exist_ok=True)
            if target.exists():
                d=np.load(target)
                if str(d['fingerprint'])!=fp:
                    raise RuntimeError(f'Stale cache {target}; source/code fingerprint changed')
                X=d['X']; view_names=d['name']
            else:
                image=open_image(ROOT/r['path'])
                if r['dataset']=='OwensLab/CommunityForensics-Small':
                    # Both classes have identical geometry and a decoded RGB -> PNG
                    # cache. Source compression history can remain; this is a control,
                    # not proof that all acquisition shortcuts have been eliminated.
                    image=ImageOps.fit(image,(512,512),method=Image.Resampling.BICUBIC)
                views=view_set(image,role,r['group'])
                X=np.stack([embed_for_score(im,model_id=cfg['clip_model_id'],variant='fuse',
                         use_forensic=True,use_tta=True,tta_jpeg_quality=70,tta_resize_scale=.5) for _,im in views])
                view_names=np.array([n for n,_ in views])
                tmp=target.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,X=X,name=view_names,fingerprint=fp)
                os.replace(tmp,target)
            chunks.append(X); n=len(X)
            ys.extend([r['label']]*n); sids.extend([r['sid']]*n); groups.extend([r['group']]*n)
            domains.extend([r['dataset']]*n); names.extend(view_names); paths.extend([r['path']]*n)
            if (i+1)%16==0 or i+1==len(records):
                elapsed=time.monotonic()-start
                print(f'{role}: {i+1}/{len(records)} images; elapsed={elapsed:.1f}s; rows={sum(len(x) for x in chunks)}',flush=True)
        tmp=base/f'{role}.tmp.npz'
        np.savez_compressed(tmp,X=np.concatenate(chunks),y=np.array(ys),sid=np.array(sids),
                           group=np.array(groups),domain=np.array(domains),name=np.array(names),path=np.array(paths),fingerprint=fp)
        os.replace(tmp,base/f'{role}.npz')
        print(f'Saved {role}: {len(ys)} views',flush=True)


if __name__=='__main__':
    main()
