"""Resumable fixed-budget V4 acquisition, with protected-content exclusion."""
from __future__ import annotations
import io
import sys
import time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image
from src.io_utils import open_image
from scripts.audit_v3 import identity
from scripts.v4_common import DATA, CACHE, REPORT, REV, SEED, read_json, write_json, preserve_v3, sha

CF = 'OwensLab/CommunityForensics-Small'
SID = 'saberzl/SID_Set'
SOURCES = [
    (CF, 40, 1, None, [('train', 64)], 4),
    (CF, 67, 1, 'guidedDiffusion', [('train', 64)], 1000),
    (CF, 60, 1, None, [('dev', 64)], 4),
    (CF, 62, 1, None, [('cal', 320)], 16),
    (CF, 100, 0, 'FFHQ', [('train', 64), ('dev', 32), ('cal', 160), ('test', 48)], 1000),
    (CF, 180, 0, 'LandscapesHQ', [('train', 64), ('dev', 32), ('cal', 160), ('test', 48)], 1000),
    (CF, 80, 1, 'StyleGANXL', [('test', 96)], 1000),
    (SID, 11, None, None, [('dev', {0: 64, 1: 32, 2: 32})], 1000),
    (SID, (13, 14), None, None, [('cal', {0: 320, 1: 160, 2: 160})], 1000),
    (SID, 15, None, None, [('test', {0: 96, 1: 48, 2: 48})], 1000),
]


def blocked_identities():
    cache_path = CACHE / 'blocked.json'
    old = read_json(ROOT / 'data/v3/manifest.json')
    protected = sorted((ROOT / 'data/demo_subset').glob('*.png'))
    if len(protected) != 2000:
        raise RuntimeError('All 2000 official reference images must be protected')
    paths = sorted({r['path'] for r in old['records']} | {p.relative_to(ROOT).as_posix() for p in protected})
    stamps = [(p, (ROOT/p).stat().st_size, (ROOT/p).stat().st_mtime_ns) for p in paths]
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached['stamps'] == [list(x) for x in stamps]:
            return cached['identities']
    legacy = read_json(ROOT/'artifacts/v3/content_identities.json')
    ids = []
    for i, (p, size, stamp) in enumerate(stamps):
        entry = legacy.get(p)
        if entry and entry['stamp'] == f'{size}:{stamp}':
            ids.append(entry['identity'])
        else:
            ids.append(list(identity(open_image(ROOT/p))))
        if (i+1) % 2000 == 0:
            print(f'Protected identity inventory {i+1}/{len(paths)}', flush=True)
    write_json(cache_path, dict(stamps=stamps, identities=ids))
    return ids


def main():
    preserve_v3()
    manifest = DATA/'manifest.json'
    protocol_sha = sha(ROOT/'docs/v4_protocol.md')
    state = read_json(manifest) if manifest.exists() else dict(
        seed=SEED, complete=False, protocol_sha256=protocol_sha, revisions=REV,
        records=[], excluded=Counter(), source_family_disjoint=True)
    if state['protocol_sha256'] != protocol_sha or state['revisions'] != REV:
        raise RuntimeError('Acquisition protocol changed after start')
    if state['complete']:
        print('Frozen V4 acquisition already complete', flush=True); return
    ids = blocked_identities()
    exact = {x[0] for x in ids}; near = [(int(p), int(d)) for _,p,d in ids]
    for row in state['records']:
        exact.add(row['group']); near.append((int(row['phash'],16),int(row['dhash'],16)))
    consumed = {r.get('source_id') for r in state['records']}
    excluded = Counter(state['excluded'])
    fs = HfFileSystem()
    def checkpoint():
        state['excluded'] = dict(excluded)
        state['counts'] = dict(Counter(r['role'] for r in state['records']))
        write_json(manifest, state)
    def accept(image, row):
        h, p, d = identity(image)
        if h in exact or any((p^pp).bit_count()<=4 and (d^dd).bit_count()<=2 for pp,dd in near):
            excluded['exact_or_near_old_protected_or_new'] += 1; return False
        target = DATA/'images'/f'{h}.png'; target.parent.mkdir(parents=True,exist_ok=True)
        image.save(target, format='PNG')
        row.update(group=h, phash=f'{p:016x}', dhash=f'{d:016x}', path=target.relative_to(ROOT).as_posix())
        exact.add(h); near.append((p,d)); consumed.add(row['source_id']); state['records'].append(row)
        if len(state['records'])%16 == 0: checkpoint()
        return True
    for dataset, shards, label, model, quotas, cap in SOURCES:
        shards = (shards,) if isinstance(shards,int) else shards
        previous = [r for r in state['records'] if r['dataset']==dataset and r['shard'] in shards]
        got = Counter((r['role'],r['sid']) for r in previous)
        models = Counter((r['role'],r['model']) for r in previous)
        def remaining():
            return [(role, sid, n) for role,q in quotas for sid,n in
                    (q.items() if isinstance(q,dict) else [(label,q)]) if got[(role,sid)] < n]
        if not remaining(): continue
        for shard in shards:
            if not remaining(): break
            print(f'Acquire {dataset} shard={shard} outstanding={remaining()}',flush=True)
            filename = f'data/HFCF_small_{shard}.parquet' if dataset==CF else f'data/validation-{shard:05d}-of-00034.parquet'
            url = f'datasets/{dataset}@{REV[dataset]}/{filename}'
            columns = ['image_data','image_name','label','model_name','architecture','nsfw_flag'] if dataset==CF else ['image','img_id','label']
            # Reopen/replay on transport failure. Accepted source IDs and content stay fixed.
            for attempt in range(4):
                try:
                    pf = pq.ParquetFile(fs.open(url,block_size=1<<20),buffer_size=1<<20,pre_buffer=False)
                    try:
                        for batch in pf.iter_batches(batch_size=16,columns=columns):
                            for raw in batch.to_pylist():
                                sid = int(raw['label'])
                                candidates = [(r,s,n) for r,s,n in remaining() if s==sid]
                                if not candidates: continue
                                role = candidates[0][0]
                                m = raw['model_name'] if dataset==CF else 'SID'
                                if dataset==CF and (raw['nsfw_flag'] or (model and m!=model) or models[(role,m)]>=cap): continue
                                source = f'{dataset}:{shard}:{m}:{raw["image_name"] if dataset==CF else raw["img_id"]}'
                                if source in consumed: continue
                                payload = raw['image_data'] if dataset==CF else raw['image']['bytes']
                                with Image.open(io.BytesIO(payload)) as im: image=im.convert('RGB')
                                row = dict(role=role,label=int(sid!=0),sid=sid,dataset=dataset,model=m,source_id=source,shard=shard)
                                if accept(image,row): got[(role,sid)]+=1; models[(role,m)]+=1
                            if not remaining(): break
                    finally: pf.close()
                    break
                except Exception as exc:
                    checkpoint()
                    print(f'Acquisition retry {attempt+1}/4: {type(exc).__name__}: {exc}',flush=True)
                    if attempt==3: raise
                    time.sleep(2**attempt)
            checkpoint()
        if remaining(): raise RuntimeError(f'Unfilled fixed quotas: {dataset} {shards} {remaining()}')
    # Replay old training only; deliberately not new originals or content-block exceptions for any other role.
    old=read_json(ROOT/'data/v3/manifest.json')['records']; rng=np.random.default_rng(SEED)
    replay=[]
    for sid in (0,1,2):
        pool=[r for r in old if r['role']=='cached_train' and r['sid']==sid]
        for i in sorted(rng.choice(len(pool),128,replace=False)):
            r=dict(pool[i]); r['role']='replay'; replay.append(r)
    state['records'].extend(replay)
    train_models={r['model'] for r in state['records'] if r['role'] in ('train','replay')}
    if 'StyleGANXL' in train_models or 'VQDiffusion' in train_models: raise RuntimeError('Held-out generator leaked')
    for a in ('train','dev','cal','test'):
        ga={r['group'] for r in state['records'] if r['role']==a}
        for b in ('train','dev','cal','test'):
            if a!=b and ga&{r['group'] for r in state['records'] if r['role']==b}: raise RuntimeError('Split overlap')
    fake_sets={role:{r['model'] for r in state['records'] if r['role']==role and r['dataset']==CF and r['label']==1} for role in ('train','dev','cal','test')}
    if any(fake_sets[a]&fake_sets[b] for a in fake_sets for b in fake_sets if a<b):
        raise RuntimeError('CF fake model appeared in more than one role')
    state.update(complete=True, protected_old_originals=len(ids), fake_models_by_role={k:sorted(v) for k,v in fake_sets.items()})
    checkpoint(); write_json(REPORT/'data_manifest.json',state)
    print('V4 source identities FROZEN',state['counts'],flush=True)


if __name__=='__main__': main()
