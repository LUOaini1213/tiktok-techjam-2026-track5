"""Read-only source/receipt audit; no encoder load, download or inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.frozen_evidence import historical_source


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check():
    checked = 0
    index = json.loads((ROOT / "docs/frozen-source/index.json").read_text(encoding="utf-8"))
    for name, record in index["files"].items():
        historical_source(name, record["recorded_sha256"])
        checked += 1
    for version in ("v3", "v4"):
        directory = ROOT / "results" / version
        lock = json.loads((directory / "evaluation_lock.json").read_text(encoding="utf-8"))
        for name, expected in lock["report_sha256"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or digest(path.read_bytes()) != expected:
                raise ValueError("Frozen report receipt changed: " + version + "/" + name)
            checked += 1
    pretest = json.loads((ROOT / "results/v4/pretest_lock.json").read_text(encoding="utf-8"))
    for name, expected in pretest["sha256"].items():
        historical_source(name, expected)
        checked += 1
    preflight = json.loads((ROOT / "results/v4/preflight.json").read_text(encoding="utf-8"))
    historical_source("scripts/verify_v4.py", preflight["release_verification_code_sha256"])
    checked += 1
    for name, expected in (("artifacts/repostguard.joblib", json.loads(
        (ROOT / "results/v3/preflight.json").read_text(encoding="utf-8"))["shipped_model_sha256"]),
        ("results/v4/repostguard.joblib", preflight["packaged_sha256"])):
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError("Frozen packaged head changed: " + name)
        checked += 1
    return {"checks": checked, "archived_source_files": len(index["files"]),
            "mode": "read_only_hashes_no_encoder_load_or_inference"}


if __name__ == "__main__":
    try:
        print(json.dumps(check(), sort_keys=True))
    except (OSError, ValueError, KeyError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
