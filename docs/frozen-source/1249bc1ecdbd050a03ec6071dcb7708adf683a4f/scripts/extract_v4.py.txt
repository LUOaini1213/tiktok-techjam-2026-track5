"""Batched, per-original checkpointing; sealed test requires a valid pretest lock."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageOps
from src.augment import EVAL_PRESETS, apply_named
from src.repost import CHAINS, apply_chain
from src.io_utils import open_image
from src.features_v4 import embed_v4_batch, V4_DIM
from scripts.v4_common import DATA,CACHE,REPORT,read_json,save_npz,source_fingerprint,preserve_v3,sha


def view_set(image, role):
    if role in ('train','replay','cal'):
        yield 'clean',image
        yield 'crop_resize_jpeg',apply_chain(image,'crop_resize_jpeg')
        return
    wanted={'clean','jpeg_30','blur_1.0','resize_0.25','noise_0.05'}
    if role=='test': wanted={n for n,_,_ in EVAL_PRESETS}
    for n,op,p in EVAL_PRESETS:
        if n in wanted: yield n,apply_named(image,op,p)
    chains=list(CHAINS) if role=='test' else ['crop_resize_jpeg']
    for n in chains: yield n,apply_chain(image,n)


def validate_lock():
    lock=read_json(REPORT/'pretest_lock.json')
    for name,digest in lock['sha256'].items():
        if sha(ROOT/name)!=digest: raise RuntimeError(f'Pretest lock changed: {name}')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--roles',default='train,replay,cal,dev')
    p.add_argument('--batch_size',type=int,default=8); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    if a.batch_size<1: raise ValueError('Positive batch size required')
    preserve_v3(); manifest=read_json(DATA/'manifest.json')
    if not manifest['complete']: raise RuntimeError('Unfrozen dataset')
    fp=source_fingerprint(DATA/'manifest.json')
    for role in a.roles.split(','):
        if role=='test': validate_lock()
        records=[r for r in manifest['records'] if r['role']==role]
        if not records: raise RuntimeError(f'Empty role {role}')
        chunks=[]; values={k:[] for k in ('y','sid','group','domain','name','path')}
        start=time.monotonic()
        for i,r in enumerate(records):
            target=CACHE/'images'/f'{role}_{r["group"]}.npz'
            if target.exists():
                with np.load(target) as d:
                    if str(d['fingerprint'])!=fp: raise RuntimeError(f'Stale cache {target}')
                    X=d['X']; names=d['name']
            else:
                image=open_image(ROOT/r['path'])
                if r['dataset']=='OwensLab/CommunityForensics-Small':
                    image=ImageOps.fit(image,(512,512),method=Image.Resampling.BICUBIC)
                pieces=[]; names=[]; pending=[]
                for name,work in view_set(image,role):
                    names.append(name); pending.append(work)
                    if len(pending)==a.batch_size:
                        pieces.append(embed_v4_batch(pending)); pending=[]
                if pending: pieces.append(embed_v4_batch(pending))
                X=np.concatenate(pieces); names=np.array(names)
                save_npz(target,X=X,name=names,fingerprint=fp)
            if X.shape[1]!=V4_DIM or not np.isfinite(X).all(): raise RuntimeError('Invalid feature cache')
            chunks.append(X); n=len(X)
            for key,item in [('y',r['label']),('sid',r['sid']),('group',r['group']),('domain',r['dataset']),('path',r['path'])]:
                values[key].extend([item]*n)
            values['name'].extend(names)
            if (i+1)%16==0 or i+1==len(records):
                elapsed=time.monotonic()-start
                print(f'{role} {i+1}/{len(records)}; {elapsed:.1f}s; ETA {(len(records)-i-1)*elapsed/(i+1):.1f}s',flush=True)
        save_npz(CACHE/f'{role}.npz',X=np.concatenate(chunks),fingerprint=fp,**{k:np.asarray(v) for k,v in values.items()})
        print(f'Saved {role}: {sum(len(x) for x in chunks)} rows',flush=True)


if __name__=='__main__': main()
