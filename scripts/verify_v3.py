"""Post-workflow deployment checks; explicit --promote preserves the archived baseline."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from src.model import load_bundle,save_bundle,predict_proba


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--promote',action='store_true'); args=p.parse_args()
    reports=ROOT/'results/v3'; logs=ROOT/'logs/v3'
    selection=json.loads((reports/'selection.json').read_text(encoding='utf-8'))
    evidence=json.loads((reports/'evaluation_lock.json').read_text(encoding='utf-8'))
    assert evidence['model_sha256']==selection['model_sha256']
    assert evidence['manifest_sha256']==selection['manifest_sha256']
    for name,digest in evidence['report_sha256'].items():
        assert sha(reports/name)==digest, f'Completed evidence changed: {name}'
    logs.mkdir(parents=True,exist_ok=True)
    candidate=ROOT/'artifacts/repostguard_v3.joblib'; shipped=ROOT/'artifacts/repostguard.joblib'
    assert sha(candidate)==selection['model_sha256'], 'Selected model changed after locking'
    assert sha(reports/'data_manifest.json')==selection['manifest_sha256'], 'Published manifest differs from frozen data'
    assert not selection['test_used_for_selection']
    bundle=load_bundle(candidate); meta=bundle['meta']
    assert meta['version']=='v3' and meta['feature_dim']==924 and meta['use_tta']
    test=np.load(ROOT/'artifacts/v3/test.npz')
    assert str(test['fingerprint'])==meta['feature_cache_fingerprint']
    assert len(np.unique(test['group']))==448 and len(test['X'])==448*19
    reference=np.load(ROOT/'artifacts/v3/demo.npz')
    assert len(np.unique(reference['group']))==400 and len(reference['X'])==400*6
    assert np.isfinite(test['X']).all() and np.isfinite(reference['X']).all()
    for required in ('evaluation.csv','paired_auc.csv','REPORT.md','comparison.png','error_analysis.json','protocol_audit.json'):
        assert (reports/required).stat().st_size>0, f'Missing final deliverable: {required}'
    if args.promote:
        # The frozen bundle inherited two documentation fields from v2. Correct
        # only those fields at packaging; preserve the exact pre-test bytes too.
        frozen=reports/'frozen_pretest.joblib'
        if frozen.exists() and sha(frozen)!=selection['model_sha256']:
            raise RuntimeError('Existing frozen archive was changed; refusing to replace it')
        shutil.copyfile(candidate,frozen)
        deployed_meta=dict(meta)
        deployed_meta['legacy_metadata']={k:meta[k] for k in ('seed','tampered_weight')}
        deployed_meta.update(seed=31009,tampered_weight=meta['selection']['tampered_weight'],
                             packaging_note='Inherited documentation fields corrected only; estimator, calibration and inference/decision policy identical to frozen_pretest.joblib')
        temporary=shipped.with_suffix('.v3.tmp.joblib')
        save_bundle(temporary,bundle['model'],deployed_meta)
        if sha(shipped) not in (sha(ROOT/'results/v2/repostguard.joblib'),selection['model_sha256'],sha(temporary)):
            raise RuntimeError('Shipped weights were changed by another task; refusing to overwrite')
        os.replace(temporary,shipped)
    deployed=load_bundle(shipped)
    estimator_hash=joblib.hash(bundle['model'],hash_name='sha1')
    assert joblib.hash(deployed['model'],hash_name='sha1')==estimator_hash, 'Estimator changed after the lock'
    for key in ('clip_model_id','feature_variant','feature_dim','use_forensic','use_tta',
                'tta_jpeg_quality','tta_resize_scale','decision_policy','encoder_snapshots'):
        assert deployed['meta'].get(key)==meta.get(key), f'Inference/policy metadata changed: {key}'
    np.testing.assert_array_equal(predict_proba(deployed,test['X']),predict_proba(bundle,test['X']))
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    with tempfile.TemporaryDirectory(prefix='repostguard-v3-') as temporary:
        folder=Path(temporary); inputs=folder/'images'; inputs.mkdir()
        samples=sorted((ROOT/'samples').glob('*'))
        samples=[s for s in samples if s.suffix.lower() in ('.jpg','.jpeg','.png','.webp')][:2]
        assert len(samples)==2, 'Two committed sample fixtures are required'
        for s in samples: shutil.copyfile(s,inputs/s.name)
        (inputs/'broken.jpg').write_bytes(b'Not an image; deliberate contract fixture')
        start=time.monotonic()
        proc=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'infer.py'),'--input_dir',str(inputs),
                             '--output',str(folder/'preds.json')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (logs/'contract.log').write_text(proc.stdout,encoding='utf-8')
        assert proc.returncode==0, proc.stdout[-4000:]
        assert proc.stdout.count('WARNING: failed to score ')==1, 'A valid fixture silently fell back instead of being scored'
        contract_seconds=time.monotonic()-start
        rows=json.loads((folder/'preds.json').read_text(encoding='utf-8'))
        assert len(rows)==3 and all(set(r)=={'image_path','pred'} for r in rows)
        assert all(math.isfinite(r['pred']) and 0<=r['pred']<=1 for r in rows)
        assert next(r for r in rows if Path(r['image_path']).name=='broken.jpg')['pred']==.5
    print('CLI contract passed: two valid images and an unreadable-file fallback',flush=True)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/','-q'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    (logs/'pytest.log').write_text(tests.stdout,encoding='utf-8')
    assert tests.returncode==0, tests.stdout[-5000:]
    # Exercise the actual current UI callback, not just an import or mocked feature.
    os.environ['HF_HUB_OFFLINE']='1'
    from PIL import Image
    import app
    app.load_weights()
    with Image.open(samples[0]) as im:
        label,score,view=app.score_image(im.copy(),100,0,0,1,1,'double_repost')
    assert np.isfinite(score) and 0<=score<=1 and view.mode=='RGB'
    assert 'Thresholds are calibration-specific' in label
    result=dict(shipped_model_sha256=sha(shipped),frozen_pretest_sha256=selection['model_sha256'],
                unchanged_estimator_sha1=estimator_hash,selected=selection['selected'],
                metadata_only_corrections=dict(seed=dict(old=meta['seed'],new=deployed['meta']['seed']),
                    tampered_weight=dict(old=meta['tampered_weight'],new=deployed['meta']['tampered_weight'])),
                cli_contract=dict(rows=3,bad_image_score=.5,wall_seconds_including_startup=contract_seconds),
                pytest=re.findall(r'\d+ passed[^\r\n]*',tests.stdout)[-1],
                ui_callback=dict(scenario='double_repost',score=score,verdict=label),
                evaluation_complete=True,test_used_for_selection=False)
    (reports/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
