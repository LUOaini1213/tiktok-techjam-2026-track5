"""Regenerate tests/golden_smoke.json from the heads CI smoke-tests.

The two files in samples/ are procedural fixtures (scripts/make_samples.py), not AIGC.
They exist to exercise the infer.py contract, and CI used to assert nothing about them
beyond 0 <= pred <= 1 -- which stayed green when the shipped scores moved across the 0.5
threshold. This script writes the current scores down so CI can assert them to 1e-3.

CI runs infer.py with two heads: the default artifacts/repostguard.joblib (top-level
"tta" / "no_tta") and the opt-in V4 artifact results/v4/repostguard.joblib ("opt_in_v4").

Run it after retraining or promoting a head, changing an encoder, or touching the feature
path:

    python scripts/make_golden_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import score_directory
from src.io_utils import enable_utf8_stdout

GOLDEN = ROOT / "tests" / "golden_smoke.json"
SAMPLES = ROOT / "samples"
DEFAULT_WEIGHTS = "artifacts/repostguard.joblib"
OPT_IN_V4_WEIGHTS = "results/v4/repostguard.joblib"


def _scores(weights: str, use_tta: bool) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as td:
        rows = score_directory(SAMPLES, Path(td) / "preds.json", weights=ROOT / weights, use_tta=use_tta)
    return {Path(r["image_path"]).name: round(float(r["pred"]), 6) for r in rows}


def main() -> None:
    enable_utf8_stdout()
    previous = json.loads(GOLDEN.read_text(encoding="utf-8"))
    payload = {
        "note": previous["note"],
        "tolerance": previous["tolerance"],
        "weights": DEFAULT_WEIGHTS,
        "tta": _scores(DEFAULT_WEIGHTS, True),
        "no_tta": _scores(DEFAULT_WEIGHTS, False),
        "opt_in_v4": {
            "weights": OPT_IN_V4_WEIGHTS,
            "tta": _scores(OPT_IN_V4_WEIGHTS, True),
            "no_tta": _scores(OPT_IN_V4_WEIGHTS, False),
        },
    }
    GOLDEN.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"Wrote {GOLDEN}")
    print(json.dumps({k: payload[k] for k in ("tta", "no_tta", "opt_in_v4")}, indent=2))


if __name__ == "__main__":
    main()
