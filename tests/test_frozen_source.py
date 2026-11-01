"""Historical SHA checks reject corruption of either archived or current code."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from src.frozen_evidence import ROOT, SOURCE_COMMIT, historical_source


def test_archive_is_byte_identical_to_recorded_git_source():
    index = json.loads((ROOT / "docs/frozen-source/index.json").read_text())
    for name, record in index["files"].items():
        # Local history may be shallow in CI; the immutable per-file receipt is
        # always checked, and this independent Git comparison runs when available.
        original = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=ROOT,
                                  capture_output=True)
        archived = historical_source(name, record["recorded_sha256"])
        if original.returncode == 0:
            assert archived == original.stdout
        assert hashlib.sha256(archived).hexdigest() == record["recorded_sha256"]


@pytest.fixture
def fixture_root(tmp_path):
    name = "src/features.py"
    current = b"current source\n"; original = b"historical source\r\n"
    path = tmp_path / name; path.parent.mkdir(); path.write_bytes(current)
    snapshot_name = f"docs/frozen-source/{SOURCE_COMMIT}/{name}.txt"
    snapshot = tmp_path / snapshot_name; snapshot.parent.mkdir(parents=True); snapshot.write_bytes(original)
    record = {"snapshot": snapshot_name, "recorded_sha256": hashlib.sha256(original).hexdigest(),
              "runtime_sha256": hashlib.sha256(current).hexdigest()}
    index = {"schema": 1, "source_commit": SOURCE_COMMIT, "files": {name: record}}
    (tmp_path / "docs/frozen-source/index.json").write_text(json.dumps(index))
    return tmp_path, name, record


def test_current_and_historical_source_are_separately_guarded(fixture_root):
    root, name, record = fixture_root
    assert historical_source(name, record["recorded_sha256"], root=root) == b"historical source\r\n"
    (root / name).write_bytes(b"changed runtime\n")
    with pytest.raises(ValueError, match="Current runtime"):
        historical_source(name, record["recorded_sha256"], root=root)


def test_snapshot_corruption_and_wrong_receipt_fail_closed(fixture_root):
    root, name, record = fixture_root
    with pytest.raises(ValueError, match="receipt mismatch"):
        historical_source(name, "f" * 64, root=root)
    (root / record["snapshot"]).write_bytes(b"changed historical source\n")
    with pytest.raises(ValueError, match="receipt mismatch"):
        historical_source(name, record["recorded_sha256"], root=root)


def test_snapshot_path_cannot_escape_or_change_recorded_commit(fixture_root):
    root, name, record = fixture_root
    path = root / "docs/frozen-source/index.json"
    index = json.loads(path.read_text()); index["files"][name]["snapshot"] = "../outside.txt"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="snapshot path"):
        historical_source(name, root=root)
    with pytest.raises(ValueError, match="escapes"):
        historical_source("../outside.txt", root=root)
