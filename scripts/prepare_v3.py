"""Freeze a provenance manifest before model selection; never load WildFake for training.

CommunityForensics-Small: CC-BY-NC-SA-4.0, non-commercial research. Only FFHQ and
LandscapesHQ reals are used; all COCO images are excluded to protect the official demo.
VQDiffusion is a completely held-out generator family. SID test comes from shard 10,
outside the prefix used by the historical subset. Local source files are never replaced.
"""
from __future__ import annotations
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from PIL import Image
from src.data import records_from_folders
from src.io_utils import open_image

OUT = ROOT / 'data/v3'
MANIFEST = OUT / 'manifest.json'
SEED = 31009


def digest(image):
    arr = np.asarray(image.convert('RGB'))
    return hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()


def dhash(image):
    arr = np.asarray(image.convert('L').resize((9, 8)))
    return int(''.join('1' if x else '0' for x in (arr[:, 1:] > arr[:, :-1]).ravel()), 2)


def write_manifest(rows, audit, revisions, complete=False):
    payload = dict(seed=SEED, complete=complete, records=rows, audit=audit, revisions=revisions,
                   license='CommunityForensics-Small: CC-BY-NC-SA-4.0; SID-Set: CC-BY-4.0',
                   protocol='Frozen before CV. No WildFake/COCO data in train, dev or calibration. VQDiffusion test only.')
    tmp = MANIFEST.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(MANIFEST)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding='utf-8')).get('complete'):
        print('Frozen manifest already complete; refusing to reassign roles', flush=True)
        return
    rows, seen, near = [], set(), []
    existing = records_from_folders(ROOT / 'data/sid_subset')
    train = [r for r in existing if r['split'] == 'train']
    hashes = [hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in train]
    unique = {}
    for i, h in enumerate(hashes):
        unique.setdefault(h, i)
    audit = dict(legacy_train_files=len(train), legacy_train_unique_files=len(unique),
                 legacy_train_duplicate_files=len(train)-len(unique))
    # Manifest includes cache indices so filtering cannot reorder the cached image blocks.
    for h, idx in unique.items():
        r = train[idx]
        rows.append(dict(path=Path(r['path']).relative_to(ROOT).as_posix(), label=r['label'],
                         sid=int(Path(r['path']).stem.split('_')[0]), role='cached_train',
                         group=h, cache_index=idx, dataset='SID-Set legacy', model='SID'))
    old_cal = set(json.loads((ROOT/'artifacts/cal_images.json').read_text(encoding='utf-8')))
    train_hashes = set(hashes)
    overlap, val_dupes = 0, 0
    all_val_hashes = set()
    candidates = {0: [], 1: []}
    for r in existing:
        if r['split'] != 'val':
            continue
        h = hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
        if h in train_hashes:
            overlap += 1
            continue
        if h in all_val_hashes:
            val_dupes += 1
            continue
        all_val_hashes.add(h)
        if str(Path(r['path'])) in old_cal:
            candidates[r['label']].append(r)
    for label in (0, 1):
        for r in candidates[label][:150]:
            image = open_image(Path(r['path']))
            h = digest(image)
            rows.append(dict(path=Path(r['path']).relative_to(ROOT).as_posix(), label=label,
                             sid=int(Path(r['path']).stem.split('_')[0]), role='cal', group=h,
                             dataset='SID-Set legacy cal', model='SID'))
            seen.add(h)
            near.append(dhash(image))
    audit.update(legacy_train_val_overlap_files=overlap, legacy_val_duplicates=val_dupes)
    # A small replay of existing train images teaches chains without re-extracting 96k rows.
    rng = random.Random(SEED)
    for label in (0, 1):
        indices = [idx for idx in unique.values() if train[idx]['label'] == label]
        rng.shuffle(indices)
        for idx in indices[:96]:
            r = train[idx]
            rows.append(dict(path=Path(r['path']).relative_to(ROOT).as_posix(), label=label,
                             sid=int(Path(r['path']).stem.split('_')[0]), role='train',
                             group=hashes[idx], dataset='SID-Set replay', model='SID'))
    api, fs = HfApi(), HfFileSystem()
    revisions = {d: api.dataset_info(d).sha for d in ('OwensLab/CommunityForensics-Small', 'saberzl/SID_Set')}
    def accept(image, row):
        h, ph = digest(image), dhash(image)
        if h in seen or any((ph ^ x).bit_count() <= 2 for x in near):
            return False
        seen.add(h); near.append(ph)
        row['group'] = h
        row['dhash'] = f'{ph:016x}'
        path = OUT / 'images' / (h + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format='PNG')
        row['path'] = path.relative_to(ROOT).as_posix()
        rows.append(row)
        if len(rows) % 32 == 0:
            write_manifest(rows, audit, revisions)
        return True

    # Buffered Parquet reading streams pages from a single very large row group. Do
    # not read the whole image_data column, which can exceed RAM by several GB.
    sources = [
        (0, 1, None, [('train',128), ('dev',32), ('cal',64)]),
        (69, 1, 'StyleGAN2', [('train',128), ('dev',32), ('cal',64)]),
        (85, 1, 'BigGAN', [('train',128), ('dev',32), ('cal',64)]),
        (95, 0, 'FFHQ', [('train',192), ('dev',48), ('cal',256), ('test',48)]),
        (160, 0, 'LandscapesHQ', [('train',192), ('dev',48), ('cal',256), ('test',48)]),
        (68, 1, 'VQDiffusion', [('test',96)]),
    ]
    for shard, label, model, quotas in sources:
        prior = [r for r in previous.get('records', [])
                 if r.get('dataset')=='OwensLab/CommunityForensics-Small' and r.get('shard')==shard]
        got = Counter(r['role'] for r in prior)
        model_count = Counter((r['role'],r['model']) for r in prior)
        for r in prior:
            rows.append(r); seen.add(r['group']); near.append(int(r['dhash'],16))
        if all(got[r] == n for r,n in quotas):
            print(f'Resumed complete source {shard}: {dict(got)}',flush=True)
            continue
        print(f'Community Forensics shard={shard} label={label} roles={quotas}', flush=True)
        path = f'datasets/OwensLab/CommunityForensics-Small@{revisions["OwensLab/CommunityForensics-Small"]}/data/HFCF_small_{shard}.parquet'
        pf = pq.ParquetFile(fs.open(path, block_size=1<<20), buffer_size=1<<20, pre_buffer=False)
        for batch in pf.iter_batches(batch_size=16, columns=['image_data','image_name','label','model_name','architecture','nsfw_flag']):
            for raw in batch.to_pylist():
                if int(raw['label']) != label or (model and raw['model_name'] != model) or raw['nsfw_flag']:
                    continue
                role = next((r for r,n in quotas if got[r] < n), None)
                if role is None:
                    break
                # The LatDiff shard contains many generators; cap each to avoid one
                # model dominating any role. Other sources are generator-held-out by design.
                key = (role, raw['model_name'])
                if model is None and model_count[key] >= 16:
                    continue
                image = Image.open(io.BytesIO(raw['image_data'])).convert('RGB')
                row = dict(label=label, sid=label, role=role, dataset='OwensLab/CommunityForensics-Small',
                           model=raw['model_name'], architecture=raw['architecture'],
                           source_id=f'{shard}:{raw["model_name"]}:{raw["image_name"]}', shard=shard)
                if accept(image, row):
                    got[role] += 1; model_count[key] += 1
            if all(got[r] >= n for r,n in quotas):
                break
        pf.close()
        print(f'  accepted={dict(got)} models={len(model_count)}', flush=True)
        if any(got[r] != n for r,n in quotas):
            raise RuntimeError(f'Short source {shard}: {got}; expected {quotas}')
        write_manifest(rows, audit, revisions)

    # Fresh SID images, outside every old download prefix. Match the old storage
    # encoding only for the overlap check; the new cache itself is lossless PNG.
    print('Fresh SID test: validation shard 10', flush=True)
    path = f'datasets/saberzl/SID_Set@{revisions["saberzl/SID_Set"]}/data/validation-00010-of-00034.parquet'
    pf = pq.ParquetFile(fs.open(path, block_size=1<<20), buffer_size=1<<20, pre_buffer=False)
    quotas, got = {0:128, 1:64, 2:64}, Counter()
    blocked = train_hashes | all_val_hashes
    for batch in pf.iter_batches(batch_size=16, columns=['image','img_id','label']):
        for raw in batch.to_pylist():
            sid = int(raw['label'])
            if got[sid] >= quotas[sid]:
                continue
            payload = raw['image']['bytes']
            image = Image.open(io.BytesIO(payload)).convert('RGB')
            buf = io.BytesIO(); image.save(buf,format='JPEG',quality=95)
            if hashlib.sha256(buf.getvalue()).hexdigest() in blocked:
                continue
            if accept(image, dict(label=int(sid != 0), sid=sid, role='test', dataset='SID-Set fresh',
                                  model='SID', source_id=raw['img_id'], shard=10)):
                got[sid] += 1
        if all(got[k] >= v for k,v in quotas.items()):
            break
    pf.close()
    if any(got[k] < v for k,v in quotas.items()):
        raise RuntimeError(f'Fresh SID test short: {got}')
    train_models = {r['model'] for r in rows if r['role'] in ('train','cached_train')}
    assert 'VQDiffusion' not in train_models
    audit['counts_by_role'] = dict(Counter(r['role'] for r in rows))
    audit['new_training_generators'] = sorted({r['model'] for r in rows if r['role']=='train' and r['label']==1 and r['model']!='SID'})
    write_manifest(rows, audit, revisions, complete=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
