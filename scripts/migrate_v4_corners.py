"""Before fitting only: recompute V4 corners, preserve every frozen neural feature.

Per-image originals are backed up. The first 1692 dimensions are never computed,
modified or relabeled. Only the 56 corner dimensions and source fingerprint change.
No test image features exist or are touched by this pre-fit engineering migration.
"""
from __future__ import annotations
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageOps
from src.features_v4 import corner_summary
from src.io_utils import open_image
from scripts.extract_v4 import view_set
from scripts.v4_common import DATA,CACHE,REPORT,read_json,write_json,save_npz,source_fingerprint,sha,preserve_v3,awake_during_job


def main():
    preserve_v3()
    if (REPORT/'pretest_lock.json').exists(): raise RuntimeError('Migration is prohibited after model selection lock')
    old=read_json(CACHE/'corner_migration_before.json')
    if old['manifest_sha256']!=sha(DATA/'manifest.json'): raise RuntimeError('Source identities changed')
    fp=source_fingerprint(DATA/'manifest.json')
    rows={(r['role'],r['group']):r for r in read_json(DATA/'manifest.json')['records']}
    output=REPORT/'corner_migration.json'
    report=read_json(output) if output.exists() else dict(complete=False,old_fingerprint=old['old_fingerprint'],
            new_fingerprint=fp,old_feature_source_sha256=old['old_feature_source_sha256'],
            old_source_commit='7f1c37e',manifest_sha256=old['manifest_sha256'],
            neural_and_global_prefix_dimensions_preserved=1692,corner_dimensions_recomputed=56,
            algorithm='Vectorized identical 8x8 DCT bands and same native residual/FFT/color statistics; engineering only, before any V4 fitting/test extraction',
            files=[],max_absolute_corner_difference=0.,exactly_equal_corner_files=0)
    if report['new_fingerprint']!=fp: raise RuntimeError('Migration source changed mid-run')
    done={r['file'] for r in report['files']}; start=time.monotonic()
    files=[p for p in sorted((CACHE/'images').glob('*.npz')) if '.tmp.' not in p.name]
    for i,path in enumerate(files):
        role,group=path.stem.split('_',1)
        if role=='test': raise RuntimeError('Sealed test must not exist before selection')
        row=rows[(role,group)]
        with np.load(path) as z: d={k:z[k] for k in z.files}
        if str(d['fingerprint'])==fp:
            if path.name not in done: raise RuntimeError('New-tag cache has no migration record')
            continue
        if str(d['fingerprint'])!=old['old_fingerprint']: raise RuntimeError(f'Unexpected old cache provenance: {path.name}')
        backup=CACHE/'legacy_corners'/path.name; backup.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(path)
        if not backup.exists(): shutil.copy2(path,backup)
        if sha(backup)!=digest: raise RuntimeError('Existing migration backup differs from original')
        image=open_image(ROOT/row['path'])
        if row['dataset']=='OwensLab/CommunityForensics-Small':
            image=ImageOps.fit(image,(512,512),method=Image.Resampling.BICUBIC)
        corners=[]; names=[]
        for name,work in view_set(image,role): names.append(name); corners.append(corner_summary(work))
        if not np.array_equal(d['name'],np.array(names)): raise RuntimeError('Transformed view sequence changed')
        corner=np.stack(corners); difference=float(np.max(np.abs(d['X'][:,1692:]-corner)))
        np.testing.assert_allclose(d['X'][:,1692:],corner,atol=2e-6,rtol=2e-6)
        prefix=d['X'][:,:1692].copy(); equal=np.array_equal(d['X'][:,1692:],corner)
        d['X']=d['X'].copy(); d['X'][:,1692:]=corner
        np.testing.assert_array_equal(prefix,d['X'][:,:1692])
        d['fingerprint']=fp; save_npz(path,**d)
        report['files'].append(dict(file=path.name,old_sha256=digest,new_sha256=sha(path),max_corner_difference=difference))
        report['max_absolute_corner_difference']=max(report['max_absolute_corner_difference'],difference)
        report['exactly_equal_corner_files']+=int(equal)
        write_json(output,report)
        if (i+1)%32==0: print(f'Corner migration {i+1}/{len(files)}; {time.monotonic()-start:.1f}s; maxdiff {report["max_absolute_corner_difference"]}',flush=True)
    report['complete']=True; report['files_migrated']=len(report['files']); write_json(output,report)
    preserve_v3(); print('Corner migration complete; neural/global prefix unchanged',report['files_migrated'],flush=True)


if __name__=='__main__':
    with awake_during_job(): main()
