"""The published experiment stays byte-verifiable after a fresh Git checkout."""
from pathlib import Path
import hashlib
import json

import joblib

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'results/v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))


def test_published_report_and_manifest_checksums():
    selection = read_json('selection.json')
    evidence = read_json('evaluation_lock.json')
    assert not selection['test_used_for_selection']
    assert not evidence['test_used_for_selection']
    assert evidence['model_sha256'] == selection['model_sha256']
    assert sha(REPORTS / 'frozen_pretest.joblib') == selection['model_sha256']
    assert sha(REPORTS / 'data_manifest.json') == selection['manifest_sha256']
    for name, digest in evidence['report_sha256'].items():
        assert sha(REPORTS / name) == digest, name


def test_published_source_bytes_match_feature_and_training_provenance():
    meta = joblib.load(ROOT / 'artifacts/repostguard.joblib')['meta']
    sources = ('src/features.py', 'src/repost.py', 'src/augment.py',
               'scripts/extract_v3.py')
    payload = (REPORTS / 'data_manifest.json').read_bytes()
    payload += b''.join((ROOT / name).read_bytes() for name in sources)
    assert hashlib.sha256(payload).hexdigest() == meta['feature_cache_fingerprint']
    assert sha(ROOT / 'scripts/train_v3.py') == meta['training_code_sha256']


def test_packaged_head_preserves_the_frozen_estimator_and_policy():
    frozen = joblib.load(REPORTS / 'frozen_pretest.joblib')
    shipped = joblib.load(ROOT / 'artifacts/repostguard.joblib')
    assert joblib.hash(frozen['model']) == joblib.hash(shipped['model'])
    for key in ('decision_policy', 'encoder_snapshots', 'feature_variant',
                'use_tta', 'tta_jpeg_quality', 'tta_resize_scale'):
        assert frozen['meta'][key] == shipped['meta'][key], key
    assert shipped['meta']['seed'] == 31009
    assert shipped['meta']['tampered_weight'] == shipped['meta']['selection']['tampered_weight']
    assert read_json('preflight.json')['shipped_model_sha256'] == sha(ROOT / 'artifacts/repostguard.joblib')
