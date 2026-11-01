"""Deployment parameters come from the artifact, not whichever config is nearby."""
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
import infer


def test_inference_uses_artifact_tta_and_invalid_scores_fall_back(tmp_path,monkeypatch):
    Image.new('RGB',(20,24)).save(tmp_path/'a.png')
    bundle={'model':object(),'meta':dict(feature_variant='fuse',feature_dim=924,
             use_forensic=True,use_tta=True,tta_jpeg_quality=50,tta_resize_scale=.25)}
    monkeypatch.setattr(infer,'load_bundle',lambda p:bundle)
    seen=[]
    def embed(image,**kwargs):
        seen.append(kwargs)
        return np.full(924,np.nan,dtype=np.float32)
    monkeypatch.setattr(infer,'embed_for_score',embed)
    out=tmp_path/'out.json'
    rows=infer.score_directory(tmp_path,out,cpu_threads=1)
    assert seen[0]['tta_jpeg_quality']==50 and seen[0]['tta_resize_scale']==.25
    assert len(rows)==1 and rows[0]['pred']==.5
    assert set(rows[0])=={'image_path','pred'}
    assert rows==json.loads(out.read_text(encoding='utf-8'))


def test_invalid_cpu_threads_are_rejected_before_loading_weights(tmp_path):
    with pytest.raises(ValueError,match='positive'):
        infer.score_directory(tmp_path,tmp_path/'out.json',cpu_threads=0)


def test_gradio_component_contract_builds_without_starting_a_server(monkeypatch):
    import app
    monkeypatch.setattr(app,'load_weights',lambda:None)
    demo=app.build_demo()
    assert len(demo.input_components)==7 and len(demo.output_components)==3
