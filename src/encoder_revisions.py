"""Checkpoint identity for runtime loading and newly extracted feature caches.

No Hub lookup occurs here. Missing metadata is explicitly legacy/unpinned; an
existing snapshot map is never silently replaced by a model's default branch.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import re
import warnings

DINO_MODEL_ID = "facebook/dinov2-small"
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_ACTIVE = ContextVar("repostguard_encoder_snapshots", default=None)


class LegacyEncoderWarning(UserWarning):
    """The caller has no recorded immutable encoder identity."""


class EncoderLoadError(RuntimeError):
    """A shared encoder could not load; this is not a broken input image."""


def validate_snapshots(snapshots, model_id, variant="fuse", *, allow_legacy=True):
    if snapshots is None:
        if not allow_legacy:
            raise ValueError("Encoder snapshots are required; re-extract caches with pinned revisions")
        warnings.warn("Legacy encoder metadata: loading the default Hub revision; "
                      "this run is unpinned and cannot reproduce a recorded checkpoint identity.",
                      LegacyEncoderWarning, stacklevel=2)
        return None
    if not isinstance(snapshots, Mapping) or not snapshots:
        raise ValueError("encoder_snapshots must be a nonempty model-id to immutable SHA map")
    result = {}
    for key, revision in snapshots.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError("encoder_snapshots requires nonempty model IDs and full lowercase 40-hex commit SHAs")
        result[key] = revision
    required = {model_id}
    if variant in ("dino", "fuse", "v4_global", "v4_local"):
        required.add(DINO_MODEL_ID)
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError("Missing encoder snapshots for: " + ", ".join(missing))
    return result


@contextmanager
def encoder_scope(snapshots, model_id, variant="fuse"):
    # Context-local state supports nested calls without process-global mutation.
    checked = validate_snapshots(snapshots, model_id, variant)
    token = _ACTIVE.set(checked)
    try:
        yield checked
    finally:
        _ACTIVE.reset(token)


def active_revision(model_id):
    snapshots = _ACTIVE.get()
    if snapshots is None:
        warnings.warn("Legacy encoder loader: default Hub revision is unpinned.",
                      LegacyEncoderWarning, stacklevel=2)
        return None
    if model_id not in snapshots:
        raise ValueError("Missing encoder snapshot for: " + model_id)
    return snapshots[model_id]


def snapshot_json(snapshots):
    return json.dumps(snapshots, sort_keys=True, separators=(",", ":"))


def cache_snapshots(cache, model_id, variant="fuse"):
    """New training must not infer cache identity from refs/main on this machine."""
    if "encoder_snapshots" not in cache:
        raise ValueError("Feature cache lacks encoder snapshots; re-extract rather than guessing its checkpoint identity")
    try:
        value = json.loads(str(cache["encoder_snapshots"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid feature-cache encoder snapshots") from exc
    return validate_snapshots(value, model_id, variant, allow_legacy=False)


def matching_cache_snapshots(caches, model_id, variant="fuse"):
    maps = [cache_snapshots(cache, model_id, variant) for cache in caches]
    if not maps or any(value != maps[0] for value in maps[1:]):
        raise ValueError("Feature caches use different encoder snapshots")
    return maps[0]


def feature_fingerprint(payload, snapshots):
    # A new namespace prevents versioned caches from impersonating legacy ones.
    return hashlib.sha256(b"repostguard-pinned-encoders-v1\0" + payload +
                          snapshot_json(snapshots).encode("utf-8")).hexdigest()
