"""Audit split/protected-benchmark identities without consulting prediction scores."""
from __future__ import annotations
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.audit_v3 import identity
from src.io_utils import open_image


def main():
    path=ROOT/'data/v3/manifest.json'
    m=json.loads(path.read_text(encoding='utf-8')); rows=m['records']
    assert m['complete'] and m['audit']['content_audit_complete']
    groups={}
    for row in rows:
        role='train' if row['role']=='cached_train' else row['role']
        previous=groups.setdefault(row['group'],(role,row['label']))
        assert previous==(role,row['label']), 'Identity crosses a split or label'
        if role in ('train','cal','dev'):
            assert row['model']!='VQDiffusion', 'Held-out generator entered development'
            assert 'COCO' not in row['model'] and 'WildFake' not in row['dataset']
        assert not Path(row['path']).is_absolute(), 'Manifest must use portable relative paths'
    cache=json.loads((ROOT/'artifacts/v3/content_identities.json').read_text(encoding='utf-8'))
    training_paths={r['path'] for r in rows if r['role'] in ('train','cached_train')}
    train=[cache[p]['identity'] for p in training_paths]
    tampered_paths={r['path'] for r in rows if r['role'] in ('train','cached_train') and r['sid']==2}
    tampered=[(p,cache[p]['identity']) for p in tampered_paths]
    protected=sorted((ROOT/'data/demo_subset').glob('*.png'))
    if not protected: raise RuntimeError('Protected reference cache is missing')
    matches=[]; parent_candidates=[]
    for p in protected:
        sha,ph,dh=identity(open_image(p))
        if any(sha==h or ((ph^pp).bit_count()<=4 and (dh^dd).bit_count()<=2) for h,pp,dd in train):
            matches.append(p.relative_to(ROOT).as_posix())
        for source,(_,pp,dd) in tampered:
            if (ph^pp).bit_count()<=12 and (dh^dd).bit_count()<=10:
                parent_candidates.append(dict(reference=p.relative_to(ROOT).as_posix(),training=source,
                                              phash_distance=(ph^pp).bit_count(),dhash_distance=(dh^dd).bit_count()))
    if matches: raise RuntimeError(f'Protected reference near matches to training: {matches[:10]}')
    result=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                counts_by_role=dict(Counter(r['role'] for r in rows)),
                unique_training_originals=sum(role=='train' for role,_ in groups.values()),
                protected_reference_images_checked=len(protected),protected_training_matches=0,
                identity_rule='Decoded RGB SHA256, or pHash distance <=4 AND dHash distance <=2',
                new_aigc_training_generators=sorted({r['model'] for r in rows if r['role']=='train' and r['label']==1 and r['model']!='SID'}),
                possible_parent_matches=len(parent_candidates),possible_parent_examples=parent_candidates[:12],
                source_scope_correction='CF real sources exclude COCO and no protected reference is deliberately sampled. SID uses COCO captions and some COCO-derived edits. The initial frozen manifest wording about no COCO data was broader than the provenance supports. Parent-photo disjointness is not certified.',
                parent_check_rule='Exploratory SID-tampered comparison: pHash <=12 AND dHash <=10; candidates are not proof of a shared parent. Zero candidates would not prove all parents are disjoint.',
                note='File/near-duplicate checks are not a proof against all semantic duplicates or unknown encoder-pretraining overlap.')
    dest=ROOT/'results/v3'; dest.mkdir(parents=True,exist_ok=True)
    (dest/'protocol_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
