"""Resolve archived source bytes against old receipts and the current runtime.

Snapshots are audit data (.py.txt), never imported or executed. This registry does
not establish performance for the new runtime and cannot rewrite an old receipt.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "1249bc1ecdbd050a03ec6071dcb7708adf683a4f"


def historical_source(name, expected_digest=None, *, root=ROOT):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence source path escapes repository")
    current = path.read_bytes()
    digest = hashlib.sha256(current).hexdigest()
    index = json.loads((root / "docs/frozen-source/index.json").read_text(encoding="utf-8"))
    if index.get("schema") != 1 or index.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Invalid frozen-source registry")
    record = index["files"].get(name)
    if record is None:
        if expected_digest is not None and digest != expected_digest:
            raise ValueError("Locked source changed: " + name)
        return current
    if digest != record["runtime_sha256"]:
        raise ValueError("Current runtime source changed: " + name)
    snapshot_name = f"docs/frozen-source/{SOURCE_COMMIT}/{name}.txt"
    if record["snapshot"] != snapshot_name:
        raise ValueError("Invalid frozen-source snapshot path: " + name)
    snapshot = (root / snapshot_name).resolve()
    if not snapshot.is_relative_to(root):
        raise ValueError("Frozen snapshot escapes repository")
    raw = snapshot.read_bytes()
    saved_digest = hashlib.sha256(raw).hexdigest()
    if saved_digest != record["recorded_sha256"] or (expected_digest is not None and saved_digest != expected_digest):
        raise ValueError("Frozen-source receipt mismatch: " + name)
    return raw
