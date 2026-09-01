"""Score val SID tampered files (2_*.jpg) with the shipped inference path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import score_directory
from src.config import load_config
from src.io_utils import enable_utf8_stdout


def main() -> None:
    enable_utf8_stdout()
    cfg = load_config()
    folder = cfg["data_dir"] / "val" / "aigc"
    out = cfg["results_dir"] / "tampered_scores.json"
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    rows = score_directory(folder, out)
    tamp = [r for r in rows if Path(r["image_path"]).name.startswith("2_")]
    out.write_text(json.dumps(tamp, indent=2, ensure_ascii=False), encoding="utf-8")
    preds = [r["pred"] for r in tamp]
    mean = float(sum(preds) / len(preds)) if preds else float("nan")
    n_hi = sum(1 for p in preds if p > 0.5)
    print(json.dumps({"n": len(preds), "mean": mean, "n_gt_0.5": n_hi, "path": str(out)}))


if __name__ == "__main__":
    main()
