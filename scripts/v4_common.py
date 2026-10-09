"""Portable, atomic V4 experiment bookkeeping; never mutate frozen V3 evidence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/v4'
CACHE = ROOT / 'artifacts/v4'
REPORT = ROOT / 'results/v4'
SEED = 41009
REV = {'OwensLab/CommunityForensics-Small': '6c539a534c07917307c381f5af4053c6091b5278',
       'saberzl/SID_Set': 'dc03ead57929879319ce30a82bfcfb8d317b10bd'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    os.replace(temp, path)


def save_npz(destination, **values):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temp, **values); os.replace(temp, destination)


def source_fingerprint(manifest):
    sources = ('src/features.py', 'src/augment.py', 'src/repost.py',
               'src/features_v4.py', 'scripts/extract_v4.py')
    return hashlib.sha256(Path(manifest).read_bytes() +
        b''.join((ROOT / name).read_bytes() for name in sources)).hexdigest()


def preserve_v3():
    evidence = read_json(ROOT / 'results/v3/evaluation_lock.json')
    for name, digest in evidence['report_sha256'].items():
        if sha(ROOT / 'results/v3' / name) != digest:
            raise RuntimeError(f'Frozen V3 evidence changed: {name}')
    if sha(ROOT / 'artifacts/repostguard.joblib') != read_json(ROOT / 'results/v3/preflight.json')['shipped_model_sha256']:
        raise RuntimeError('Default V3 artifact changed; V4 experiments must not replace it')
    return sha(ROOT / 'artifacts/repostguard.joblib')
