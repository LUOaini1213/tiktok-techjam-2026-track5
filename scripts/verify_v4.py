"""Check experimental release integrity and real CLI/UI; never overwrite V3."""
from __future__ import annotations
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.model import load_bundle,save_bundle
from src.model_v4 import GlobalPadAdapter
from src.io_utils import open_image,list_images
from src.score import embed_for_score
from scripts.extract_v4 import validate_lock
from scripts.v4_common import DATA,CACHE,REPORT,read_json,write_json,preserve_v3,sha


def release_meta(meta, base_meta):
    """Correct retained-classifier provenance only; frozen estimator/policy stay intact."""
    result=dict(meta)
    if meta['v4_head']=='v3_recal':
        result.update(
            calibration_selection_code_sha256=meta['training_code_sha256'],
            training_code_sha256=base_meta['training_code_sha256'],
            training_data=base_meta['training_data'],
            tampered_weight=base_meta['tampered_weight'],
            classifier_training_seed=base_meta['seed'],
            classifier_training_manifest_sha256=base_meta['manifest_sha256'],
            classifier_selection=base_meta['selection'],
            weighting='Retained V3 grouped consistency rows and tampered weight; NOT the rejected V4 source-normalized training recipe.',
            release_metadata_correction='Frozen trial metadata described the new candidate training recipe. The released artifact correctly attributes its retained V3 classifier; only V4 probability calibration and policy are new. Estimator, thresholds and feature recipe are unchanged.')
    return result


def sample_fixtures(folder):
    """Ignore .gitkeep/readmes; exercise two actual supported image files."""
    paths=list_images(folder)[:2]
    if len(paths)!=2: raise RuntimeError('Two supported sample images are required for release preflight')
    return [open_image(p) for p in paths]


def main():
    torch.set_num_threads(4); validate_lock(); default_hash=preserve_v3()
    evidence=read_json(REPORT/'evaluation_lock.json')
    for name,digest in evidence['report_sha256'].items():
        if sha(REPORT/name)!=digest: raise RuntimeError(f'Evaluation report changed {name}')
    bundle=load_bundle(REPORT/'frozen_pretest.joblib')
    needs_local='local' in bundle['meta']['v4_head']
    base=load_bundle(ROOT/'artifacts/repostguard.joblib')
    corrected_meta=release_meta(bundle['meta'],base['meta'])
    packaged=dict(model=bundle['model'],meta=corrected_meta)
    if not needs_local:
        model=GlobalPadAdapter(bundle['model'])
        with np.load(CACHE/'test.npz') as z:
            X=z['X']
            np.testing.assert_array_equal(model.predict_proba(X[:,:924]),bundle['model'].predict_proba(X))
        packaged=dict(model=model,meta=dict(corrected_meta,feature_variant='v4_global',feature_dim=924,
                    packaging_note='Global-only head ignores all 824 local columns; zero-padding adapter preserves every frozen test prediction exactly. No refit or threshold change.'))
    save_bundle(REPORT/'repostguard.joblib',**packaged)
    bundle=packaged
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    logs=ROOT/'logs/v4'; logs.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='repostguard-v4-') as tmp:
        folder=Path(tmp); inputs=folder/'images'; inputs.mkdir()
        fixtures=sample_fixtures(ROOT/'samples')
        for i,im in enumerate(fixtures): im.save(inputs/f'{i}.png')
        (inputs/'broken.jpg').write_bytes(b'Not an image: deliberate verification fixture')
        start=time.monotonic()
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'infer.py'),'--input_dir',str(inputs),
                '--output',str(folder/'preds.json'),'--weights',str(REPORT/'repostguard.joblib')],
                cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        wall=time.monotonic()-start
        (logs/'cli.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError(result.stderr)
        predictions=json.loads((folder/'preds.json').read_text(encoding='utf-8'))
        assert len(predictions)==3 and all(set(r)=={'image_path','pred'} and 0<=r['pred']<=1 for r in predictions)
        assert sum('WARNING: failed to score ' in l for l in result.stderr.splitlines())==1
        assert [r['pred'] for r in predictions if Path(r['image_path']).name=='broken.jpg']==[.5]
    manifest=read_json(DATA/'manifest.json')
    row=next(r for r in manifest['records'] if r['role']=='cal' and r['dataset']=='saberzl/SID_Set')
    image=open_image(ROOT/row['path'])
    kwargs=dict(model_id='openai/clip-vit-base-patch32',use_tta=True,use_forensic=True,tta_jpeg_quality=70,tta_resize_scale=.5,encoder_snapshots=bundle['meta'].get('encoder_snapshots'))
    deployed_variant=bundle['meta']['feature_variant']
    feat=embed_for_score(image,variant=deployed_variant,**kwargs)
    with np.load(CACHE/'images'/f'cal_{row["group"]}.npz') as z:
        clean=z['X'][np.flatnonzero(z['name']=='clean')[0]]
    np.testing.assert_allclose(feat,clean[:len(feat)],rtol=2e-4,atol=3e-5)
    runtime_score=bundle['model'].predict_proba(feat.reshape(1,-1))[0,1]
    cache_score=bundle['model'].predict_proba(clean[:len(feat)].reshape(1,-1))[0,1]
    score_drift=float(abs(runtime_score-cache_score))
    assert score_drift<=bundle['meta']['decision_policy']['numerical_score_margin']
    latency={}
    for variant in ('fuse',deployed_variant):
        embed_for_score(image,variant=variant,**kwargs)  # warm each branch separately (no cache-thrashing comparison)
        samples=[]
        for _ in range(5):
            start=time.monotonic(); embed_for_score(image,variant=variant,**kwargs); samples.append(time.monotonic()-start)
        latency[variant]=dict(warm_seconds_mean=float(np.mean(samples)),samples=samples)
    import app
    def experimental_weights():
        app.BUNDLE=bundle; app.META=bundle['meta']
    original_loader=app.load_weights
    app.load_weights=experimental_weights
    demo=app.build_demo()
    try:
        verdict,pred,transformed=app.score_image(image,100,0,0,1,1,'crop_resize_jpeg')
        assert np.isfinite(pred) and 0<=pred<=1 and transformed.size==image.size
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        demo.launch(server_name='127.0.0.1',server_port=port,prevent_thread_lock=True,quiet=True,show_error=True)
        with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}',timeout=15) as response:
            assert response.status==200
        ui=dict(score=float(pred),verdict=verdict,http_status=200,inputs=len(demo.input_components),outputs=len(demo.output_components))
    finally:
        demo.close(); app.load_weights=original_loader; app.BUNDLE=None; app.META={}
    validate_lock(); assert preserve_v3()==default_hash
    write_json(REPORT/'preflight.json',dict(default_v3_unchanged_sha256=default_hash,
               release_verification_code_sha256=sha(Path(__file__)),
               experimental_head_sha256=sha(REPORT/'frozen_pretest.joblib'),cli_rows=3,cli_wall_seconds_including_startup=wall,
               packaged_sha256=sha(REPORT/'repostguard.joblib'),packaged_variant=deployed_variant,
               global_only_padding_exact_test_predictions=not needs_local,
               measured_single_batch_calibration_score_drift=score_drift,
               broken_fallback=.5,shared_extract_infer_features_match=True,latency=latency,ui=ui,
               default_promoted=False,note='V4 remains an explicitly selectable research artifact. V3 is preserved; this check does not guarantee real-world robustness.'))
    print('V4 release preflight passed; V3 default unchanged',flush=True)


if __name__=='__main__': main()
