"""V4 evidence is portable after publication; absence during development is explicit."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'results/v4'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_pretest_lock_and_completed_reports():
    if not (REPORT/'evaluation_lock.json').exists(): pytest.skip('V4 run has not published its final lock yet')
    lock=json.loads((REPORT/'pretest_lock.json').read_text(encoding='utf-8'))
    assert not lock['test_used_for_selection']
    for name,digest in lock['sha256'].items(): assert sha(ROOT/name)==digest,name
    evidence=json.loads((REPORT/'evaluation_lock.json').read_text(encoding='utf-8'))
    assert not evidence['test_used_for_selection']
    assert sha(REPORT/'pretest_lock.json')==evidence['pretest_lock_sha256']
    for name,digest in evidence['report_sha256'].items(): assert sha(REPORT/name)==digest,name


def test_v4_role_and_heldout_family_integrity():
    if not (REPORT/'data_manifest.json').exists(): pytest.skip('V4 acquisition not yet frozen')
    m=json.loads((REPORT/'data_manifest.json').read_text(encoding='utf-8'))
    assert m['complete'] and m['counts']==dict(train=256,dev=256,cal=1280,test=384,replay=384)
    groups={}
    for r in m['records']:
        previous=groups.setdefault(r['group'],(r['role'],r['label']))
        assert previous==(r['role'],r['label'])
        assert not Path(r['path']).is_absolute()
        if r['role']!='test': assert r['model'] not in ('StyleGANXL','VQDiffusion')
        assert 'COCO' not in r['model']
    families=m['fake_models_by_role']
    for a in families:
        for b in families:
            if a!=b: assert not set(families[a])&set(families[b])
    assert families['test']==['StyleGANXL']


def test_probability_fit_and_policy_groups_are_disjoint():
    path=REPORT/'calibration_split.json'
    if not path.exists(): pytest.skip('V4 calibration split not yet published')
    split=json.loads(path.read_text(encoding='utf-8'))
    assert len(split['probability_fit'])==len(split['policy'])==640
    assert not set(split['probability_fit'])&set(split['policy'])
