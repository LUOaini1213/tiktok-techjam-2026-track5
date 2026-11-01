"""Matched official reference scores, only after the model and thresholds are locked."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.augment import apply_named
from src.config import load_config
from src.io_utils import open_image
from src.repost import apply_chain
from src.score import embed_for_score


def main():
    selection=json.loads((ROOT/'results/v3/selection.json').read_text(encoding='utf-8'))
    if hashlib.sha256((ROOT/'artifacts/repostguard_v3.joblib').read_bytes()).hexdigest()!=selection['model_sha256']:
        raise RuntimeError('Official reference may only be evaluated after model/policy lock')
    torch.set_num_threads(4); cfg=load_config(); rng=np.random.default_rng(31009)
    samples=[]
    for label in (0,1):
        paths=sorted((ROOT/'data/demo_subset').glob(f'{label}_*.png'))
        if len(paths)<200:
            raise RuntimeError('Need 200 real and 200 fake cached official demo images; run eval_demo.py --fetch_only first')
        samples.extend((paths[i],label) for i in sorted(rng.choice(len(paths),200,replace=False)))
    protocol=dict(dataset='techjam-aigc/wildfake-eval-subset',seed=31009,per_class=200,
                  purpose='official reference only; never train/calibrate/select thresholds',
                  paths=[p.relative_to(ROOT).as_posix() for p,_ in samples])
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()+b''.join(
        (ROOT/f).read_bytes() for f in ('src/features.py','src/augment.py','src/repost.py','scripts/extract_v3_demo.py'))).hexdigest()
    target=ROOT/'artifacts/v3/demo.npz'
    if target.exists():
        if str(np.load(target)['fingerprint'])!=fingerprint:
            raise RuntimeError('Official feature cache is stale')
        return
    chunks=[]; labels=[]; groups=[]; names=[]; paths=[]
    for i,(path,label) in enumerate(samples):
        out=ROOT/'artifacts/v3/images'/f'official_{path.stem}.npz'
        if out.exists():
            d=np.load(out)
            if str(d['fingerprint'])!=fingerprint: raise RuntimeError('Stale official per-image cache')
            X=d['X']; ns=d['name']
        else:
            image=open_image(path)
            views=[('clean',image),('jpeg_30',apply_named(image,'jpeg',30)),
                   ('resize_0.25',apply_named(image,'resize',.25)),('noise_0.05',apply_named(image,'noise',.05)),
                   ('crop_resize_jpeg',apply_chain(image,'crop_resize_jpeg')),('double_repost',apply_chain(image,'double_repost'))]
            X=np.stack([embed_for_score(im,model_id=cfg['clip_model_id'],variant='fuse',use_forensic=True,use_tta=True,
                                       tta_jpeg_quality=70,tta_resize_scale=.5) for _,im in views])
            ns=np.array([n for n,_ in views]); tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,X=X,name=ns,fingerprint=fingerprint); os.replace(tmp,out)
        chunks.append(X); labels.extend([label]*len(X)); groups.extend([path.stem]*len(X))
        names.extend(ns); paths.extend([path.relative_to(ROOT).as_posix()]*len(X))
        if (i+1)%16==0: print(f'Official reference: {i+1}/{len(samples)}',flush=True)
    np.savez_compressed(target,X=np.concatenate(chunks),y=np.array(labels),group=np.array(groups),
                        name=np.array(names),path=np.array(paths),
                        domain=np.array(['WildFake reference']*len(labels)),fingerprint=fingerprint)
    (ROOT/'results/v3/demo_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')


if __name__=='__main__': main()
