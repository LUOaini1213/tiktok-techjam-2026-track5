"""Optional bounded CPU helper for the tail of a role; never assigns splits or models.

Uses the unchanged extractor's view recipe and exact fingerprint. It writes only
atomic per-image caches, never combined archives or workflow state. Starting at the
tail lets the main extractor continue at the head. Concurrent identical work is safe
because temporary filenames are process-specific and final replacement is atomic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageOps
from scripts.extract_v3 import view_set
from src.config import load_config
from src.io_utils import open_image
from src.score import embed_for_score


def main():
    p=argparse.ArgumentParser(); p.add_argument('--role',choices=['train','cal','dev','test'],required=True)
    interval=p.add_mutually_exclusive_group()
    interval.add_argument('--tail',type=int,default=600)
    interval.add_argument('--head',type=int,default=None)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads); cfg=load_config()
    manifest_path=ROOT/'data/v3/manifest.json'; m=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert m['complete'] and m['audit']['content_audit_complete']
    if args.role=='test':
        selection=json.loads((ROOT/'results/v3/selection.json').read_text(encoding='utf-8'))
        assert hashlib.sha256((ROOT/'artifacts/repostguard_v3.joblib').read_bytes()).hexdigest()==selection['model_sha256']
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==selection['manifest_sha256']
    fp=hashlib.sha256(manifest_path.read_bytes()+b''.join((ROOT/f).read_bytes() for f in ('src/features.py','src/repost.py','src/augment.py','scripts/extract_v3.py'))).hexdigest()
    records=[r for r in m['records'] if r['role']==args.role]
    if args.tail<1 or (args.head is not None and args.head<1): raise ValueError('Cache interval must be positive')
    records=(records[:args.head] if args.head is not None else records[-args.tail:])[::-1]
    start=time.monotonic()
    for i,r in enumerate(records):
        target=ROOT/'artifacts/v3/images'/f'{args.role}_{r["group"]}.npz'
        if target.exists():
            if str(np.load(target)['fingerprint'])!=fp: raise RuntimeError('Incompatible feature cache')
            continue
        image=open_image(ROOT/r['path'])
        if r['dataset']=='OwensLab/CommunityForensics-Small':
            image=ImageOps.fit(image,(512,512),method=Image.Resampling.BICUBIC)
        views=view_set(image,args.role,r['group'])
        X=np.stack([embed_for_score(im,model_id=cfg['clip_model_id'],variant='fuse',use_forensic=True,
                     use_tta=True,tta_jpeg_quality=70,tta_resize_scale=.5) for _,im in views])
        temp=target.with_suffix(f'.worker{os.getpid()}.tmp.npz')
        np.savez_compressed(temp,X=X,name=np.array([n for n,_ in views]),fingerprint=fp)
        os.replace(temp,target)
        if (i+1)%16==0: print(f'Helper {args.role}: {i+1}/{len(records)} elapsed={time.monotonic()-start:.1f}s',flush=True)


if __name__=='__main__': main()
