"""Download a SID-Set subset. Falls back to a folder layout if HF is unavailable.

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --train_per_class 4000 --val_per_class 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import enable_utf8_stdout
from src.data import subsample_sid_streaming


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_per_class", type=int, default=None)
    p.add_argument("--val_per_class", type=int, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    cfg = load_config()
    out_dir = args.out_dir or cfg["data_dir"]
    train_n = args.train_per_class or cfg["train_per_class"]
    val_n = args.val_per_class or cfg["val_per_class"]
    print(f"Streaming SID-Set into {out_dir} ({train_n}/class train, {val_n}/class val)")
    manifest = subsample_sid_streaming(
        out_dir=out_dir,
        train_per_class=train_n,
        val_per_class=val_n,
        seed=cfg["seed"],
    )
    print(json_counts(manifest))


def json_counts(manifest) -> str:
    import json

    payload = manifest.get("binary_counts") or manifest.get("counts")
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    main()
