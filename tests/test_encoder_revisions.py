"""Offline identity checks: fake loaders only, no encoder downloads/forwards."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
from PIL import Image
import pytest

import infer
from src import features, score
from src.encoder_revisions import (DINO_MODEL_ID, EncoderLoadError, LegacyEncoderWarning,
    active_revision, cache_snapshots, encoder_scope, feature_fingerprint,
    matching_cache_snapshots, snapshot_json, validate_snapshots)

CLIP = "openai/clip-vit-base-patch32"
A = {CLIP: "a" * 40, DINO_MODEL_ID: "b" * 40}
B = {CLIP: "c" * 40, DINO_MODEL_ID: "d" * 40}


@pytest.mark.parametrize("bad", [{}, [], "main", {CLIP: "main"}, {CLIP: "a" * 39},
    {CLIP: "a" * 41}, {CLIP: "A" * 40}, {CLIP: "../cache"}, {CLIP: None},
    {CLIP: "a" * 40 + "\n"}, {"": "a" * 40}])
def test_invalid_snapshot_maps_reject_without_loading(bad, monkeypatch):
    monkeypatch.setattr(features, "get_device", lambda: pytest.fail("device probe"))
    with pytest.raises(ValueError):
        with encoder_scope(bad, CLIP):
            features.load_clip()


def test_missing_required_encoder_does_not_become_legacy():
    with pytest.raises(ValueError, match="Missing encoder"):
        validate_snapshots({CLIP: "a" * 40}, CLIP, "fuse")
    assert validate_snapshots({CLIP: "a" * 40}, CLIP, "proj") == {CLIP: "a" * 40}
    with pytest.raises(ValueError, match="required"):
        validate_snapshots(None, CLIP, allow_legacy=False)


@pytest.fixture
def fake_loaders(monkeypatch):
    seen = []
    class Model:
        def eval(self): return self
        def to(self, device): return self
    class Factory:
        def __init__(self, name): self.name = name
        def from_pretrained(self, model_id, **kwargs):
            seen.append((self.name, model_id, kwargs))
            return Model()
    for name in ("CLIPProcessor", "CLIPModel", "AutoImageProcessor", "AutoModel"):
        monkeypatch.setattr(features, name, Factory(name))
    monkeypatch.setattr(features, "get_device", lambda: "cpu")
    features._load_clip.cache_clear(); features._load_dino.cache_clear()
    yield seen
    features._load_clip.cache_clear(); features._load_dino.cache_clear()


def test_same_revision_for_model_processor_and_revision_aware_cache(fake_loaders):
    with encoder_scope(A, CLIP):
        first = features.load_clip(); features.load_dino()
        assert features.load_clip() is first
    with encoder_scope(B, CLIP):
        assert features.load_clip() is not first
        features.load_dino()
    assert fake_loaders == [
        ("CLIPProcessor", CLIP, {"revision": A[CLIP]}),
        ("CLIPModel", CLIP, {"revision": A[CLIP]}),
        ("AutoImageProcessor", DINO_MODEL_ID, {"revision": A[DINO_MODEL_ID]}),
        ("AutoModel", DINO_MODEL_ID, {"revision": A[DINO_MODEL_ID]}),
        ("CLIPProcessor", CLIP, {"revision": B[CLIP]}),
        ("CLIPModel", CLIP, {"revision": B[CLIP]}),
        ("AutoImageProcessor", DINO_MODEL_ID, {"revision": B[DINO_MODEL_ID]}),
        ("AutoModel", DINO_MODEL_ID, {"revision": B[DINO_MODEL_ID]}),
    ]
    # The processor's identity is also part of the actual LRU key.
    features._load_clip(CLIP, B[CLIP], A[CLIP])
    assert fake_loaders[-2][2] == {"revision": A[CLIP]}


def test_legacy_loading_warns_and_does_not_reuse_pinned_encoder(fake_loaders):
    with encoder_scope(A, CLIP): pinned = features.load_clip()
    with pytest.warns(LegacyEncoderWarning):
        with encoder_scope(None, CLIP):
            assert features.load_clip() is not pinned
    assert fake_loaders[-2:] == [("CLIPProcessor", CLIP, {"revision": None}),
                                 ("CLIPModel", CLIP, {"revision": None})]


def test_nested_failure_restores_scope_and_parallel_calls_are_isolated():
    with encoder_scope(A, CLIP):
        with pytest.raises(RuntimeError):
            with encoder_scope(B, CLIP):
                assert active_revision(CLIP) == B[CLIP]
                raise RuntimeError("fixture")
        assert active_revision(CLIP) == A[CLIP]
    barrier = Barrier(2)
    def worker(snapshot):
        with encoder_scope(snapshot, CLIP):
            barrier.wait(timeout=5)
            return active_revision(CLIP), active_revision(DINO_MODEL_ID)
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(worker, A); second = executor.submit(worker, B)
        assert first.result() == (A[CLIP], A[DINO_MODEL_ID])
        assert second.result() == (B[CLIP], B[DINO_MODEL_ID])


@pytest.mark.parametrize("variant", ["pre", "proj", "dino", "fuse", "v4_global", "v4_local"])
def test_score_boundary_enters_scope_for_every_variant(variant, monkeypatch):
    def embed(*args):
        assert active_revision(CLIP) == A[CLIP]
        assert active_revision(DINO_MODEL_ID) == A[DINO_MODEL_ID]
        return np.zeros(4)
    monkeypatch.setattr(score, "_embed_for_score", embed)
    assert score.embed_for_score(None, CLIP, variant=variant, encoder_snapshots=A).shape == (4,)


def test_real_v4_branch_inherits_scope_without_mutating_global_loaders(monkeypatch, fake_loaders):
    from src import features_v4
    def local(image, **kwargs):
        features.load_clip(); features.load_dino()
        return np.zeros(1748)
    monkeypatch.setattr(features_v4, "embed_v4", local)
    assert score.embed_for_score(None, CLIP, variant="v4_local", encoder_snapshots=A).shape == (1748,)
    assert {row[2]["revision"] for row in fake_loaders} == set(A.values())


def test_dual_score_path_carries_snapshots(monkeypatch):
    def embed(*args):
        assert active_revision(CLIP) == A[CLIP]
        assert active_revision(DINO_MODEL_ID) == A[DINO_MODEL_ID]
        return {"fuse": np.zeros(924)}
    monkeypatch.setattr(score, "_embed_for_score_dual", embed)
    assert "fuse" in score.embed_for_score_dual(None, CLIP, encoder_snapshots=A)


def bundle(snapshot=A):
    return {"model": object(), "meta": {"clip_model_id": CLIP, "feature_variant": "fuse",
        "feature_dim": 924, "use_forensic": True, "encoder_snapshots": snapshot}}


def test_cli_bad_snapshot_fails_before_probe_embedding_and_output(tmp_path, monkeypatch):
    out = tmp_path / "out.json"; out.write_text("existing evidence")
    monkeypatch.setattr(infer, "load_bundle", lambda _: bundle({CLIP: "main"}))
    monkeypatch.setattr(infer.torch.cuda, "is_available", lambda: pytest.fail("device probe"))
    monkeypatch.setattr(infer, "embed_for_score", lambda *a, **k: pytest.fail("embedding"))
    with pytest.raises(ValueError, match="40-hex"):
        infer.score_directory(tmp_path, out)
    assert out.read_text() == "existing evidence"


def test_cli_pinned_loader_failure_is_not_successful_fallback(tmp_path, monkeypatch):
    Image.new("RGB", (8, 8)).save(tmp_path / "good.png")
    out = tmp_path / "out.json"; out.write_text("existing evidence")
    monkeypatch.setattr(infer, "load_bundle", lambda _: bundle())
    monkeypatch.setattr(infer.torch.cuda, "is_available", lambda: False)
    def fail(image, **kwargs):
        assert kwargs["encoder_snapshots"] == A
        raise EncoderLoadError("missing pinned checkpoint fixture")
    monkeypatch.setattr(infer, "embed_for_score", fail)
    with pytest.raises(EncoderLoadError): infer.score_directory(tmp_path, out)
    assert out.read_text() == "existing evidence"


def test_corrupt_image_still_has_neutral_row_with_pinned_valid_image(tmp_path, monkeypatch):
    Image.new("RGB", (8, 8)).save(tmp_path / "good.png")
    (tmp_path / "broken.jpg").write_bytes(b"broken input")
    monkeypatch.setattr(infer, "load_bundle", lambda _: bundle())
    monkeypatch.setattr(infer.torch.cuda, "is_available", lambda: False)
    def embed(image, **kwargs):
        assert kwargs["encoder_snapshots"] == A
        return np.zeros(924)
    monkeypatch.setattr(infer, "embed_for_score", embed)
    monkeypatch.setattr(infer, "score_from_embed", lambda *args: 0.7)
    rows = infer.score_directory(tmp_path, tmp_path / "out.json")
    assert sorted(row["pred"] for row in rows) == [0.5, 0.7]
    assert all(set(row) == {"image_path", "pred"} for row in rows)


def test_training_cache_identity_and_fingerprint_are_not_guessed():
    cache = {"encoder_snapshots": np.array(snapshot_json(A))}
    assert cache_snapshots(cache, CLIP) == A
    assert matching_cache_snapshots([cache, cache], CLIP) == A
    with pytest.raises(ValueError, match="lacks"):
        cache_snapshots({}, CLIP)
    with pytest.raises(ValueError, match="different"):
        matching_cache_snapshots([cache, {"encoder_snapshots": snapshot_json(B)}], CLIP)
    assert feature_fingerprint(b"same source", A) != feature_fingerprint(b"same source", B)
    assert feature_fingerprint(b"changed source", A) != feature_fingerprint(b"same source", A)


def test_model_load_failure_has_shared_encoder_error(monkeypatch):
    class Bad:
        @staticmethod
        def from_pretrained(*args, **kwargs): raise OSError("checkpoint unavailable fixture")
    monkeypatch.setattr(features, "CLIPProcessor", Bad)
    monkeypatch.setattr(features, "get_device", lambda: "cpu")
    features._load_clip.cache_clear()
    with encoder_scope(A, CLIP):
        with pytest.raises(EncoderLoadError): features.load_clip()


def test_ui_validates_before_publishing_global_bundle_and_passes_snapshots(monkeypatch):
    import app
    monkeypatch.setattr(app, "BUNDLE", None); monkeypatch.setattr(app, "META", {})
    monkeypatch.setattr(app, "load_bundle", lambda _: bundle({CLIP: "main"}))
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: pytest.fail("device probe"))
    with pytest.raises(ValueError): app.load_weights()
    assert app.BUNDLE is None and app.META == {}
    monkeypatch.setattr(app, "load_bundle", lambda _: bundle())
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: False)
    app.load_weights()
    def embed(image, **kwargs):
        assert kwargs["encoder_snapshots"] == A
        return np.zeros(924)
    monkeypatch.setattr(app, "embed_for_score", embed)
    monkeypatch.setattr(app, "score_from_embed", lambda *args: 0.5)
    assert app.score_image(Image.new("RGB", (8, 8)), 100, 0, 0, 1)[1] == 0.5


def test_shard_merge_and_partial_resume_keep_revision_identity(tmp_path, monkeypatch):
    from scripts import extract_features, extract_extra_view
    paths = []
    for i in range(2):
        path = tmp_path / f"part{i}.npz"
        np.savez(path, **{f"X_{v}": np.zeros((1, 4)) for v in features.FEATURE_VARIANTS},
                 y=np.array([i]), sid=np.array([i]), views_per_image=np.array([1]),
                 shard_start=np.array([i]), n_rows_total=np.array([2]),
                 n_images_done=np.array([1]), encoder_snapshots=snapshot_json(A))
        paths.append(path)
    for module in (extract_features, extract_extra_view):
        monkeypatch.setattr(module, "load_config", lambda: {"clip_model_id": CLIP})
        out = tmp_path / f"{module.__name__}.npz"
        module.merge(paths, out)
        with np.load(out) as cache: assert cache_snapshots(cache, CLIP) == A
        with np.load(paths[1]) as cache: changed = {key: cache[key] for key in cache.files}
        changed["encoder_snapshots"] = snapshot_json(B)
        np.savez(paths[1], **changed)
        with pytest.raises(ValueError, match="different"): module.merge(paths, out)
        changed["encoder_snapshots"] = snapshot_json(A); np.savez(paths[1], **changed)
    assert extract_features._load_partial(paths[0], 1, 2, 0, A, CLIP)[-1] == 1
    assert extract_features._load_partial(paths[0], 1, 2, 0, B, CLIP)[-1] == 0


def test_generic_training_rejects_unknown_cache_identity_before_fit(tmp_path, monkeypatch):
    from scripts import train
    import sys
    np.savez(tmp_path / "features_train.npz", X_pre=np.zeros((4, 4)), y=np.array([0, 0, 1, 1]))
    monkeypatch.setattr(train, "load_config", lambda: {"artifacts_dir": tmp_path, "clip_model_id": CLIP})
    monkeypatch.setattr(sys, "argv", ["train.py"])
    monkeypatch.setattr(train, "fit_with_training_signal", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(ValueError, match="lacks encoder snapshots"): train.main()


def test_cli_main_subprocess_rejects_bad_metadata_without_fake_success(tmp_path):
    import os
    import subprocess
    import sys
    out = tmp_path / "preds.json"; out.write_text("existing evidence")
    code = '''import sys, infer
def forbidden(*args, **kwargs): raise AssertionError("unexpected probe or embedding")
infer.load_bundle=lambda _: {"meta":{"feature_variant":"fuse","encoder_snapshots":{"openai/clip-vit-base-patch32":"main"}}}
infer.torch.cuda.is_available=forbidden
infer.embed_for_score=forbidden
sys.argv=["infer.py","--input_dir",sys.argv[1],"--output",sys.argv[2]]
infer.main()
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(out)],
        cwd=infer.ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "40-hex" in result.stderr
    assert "unexpected probe or embedding" not in result.stderr and "Wrote" not in result.stdout
    assert out.read_text() == "existing evidence"


def test_v3_training_rejects_unknown_legacy_cache_before_fit(tmp_path, monkeypatch):
    from scripts import train_v3
    base = tmp_path / "artifacts/v3"; base.mkdir(parents=True)
    data = tmp_path / "data/v3"; data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({"records": []}))
    np.savez(tmp_path / "artifacts/features_train.npz", X_fuse=np.zeros((1, 924)))
    for role in ("train", "cal", "dev"):
        np.savez(base / f"{role}.npz", encoder_snapshots=snapshot_json(A))
    monkeypatch.setattr(train_v3, "ROOT", tmp_path)
    monkeypatch.setattr(train_v3, "BASE", base)
    monkeypatch.setattr(train_v3, "REPORT", tmp_path / "report")
    monkeypatch.setattr(train_v3, "fit", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(ValueError, match="lacks encoder snapshots"): train_v3.main()


def test_v4_training_rejects_unknown_legacy_cache_before_fit(tmp_path, monkeypatch):
    from scripts import train_v4, validate_v4
    cache = tmp_path / "cache"; cache.mkdir()
    artifacts = tmp_path / "artifacts/v3"; artifacts.mkdir(parents=True)
    for role in ("cal", "dev", "replay", "train"):
        np.savez(cache / f"{role}.npz", encoder_snapshots=snapshot_json(A))
    np.savez(tmp_path / "artifacts/features_train.npz", X_fuse=np.zeros((1, 924)))
    np.savez(artifacts / "train.npz", encoder_snapshots=snapshot_json(A))
    monkeypatch.setattr(train_v4, "ROOT", tmp_path)
    monkeypatch.setattr(train_v4, "CACHE", cache)
    monkeypatch.setattr(train_v4, "REPORT", tmp_path / "report")
    monkeypatch.setattr(train_v4, "preserve_v3", lambda: "fixture")
    monkeypatch.setattr(validate_v4, "validate", lambda: None)
    monkeypatch.setattr(train_v4, "fit_head", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(ValueError, match="lacks encoder snapshots"): train_v4.main()
