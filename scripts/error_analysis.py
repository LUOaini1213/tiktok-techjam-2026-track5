from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.error_analysis import collect_errors
from src.io_utils import enable_utf8_stdout

if __name__ == "__main__":
    enable_utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8, help="Top-k FP/FN examples to keep")
    p.add_argument("--max_images", type=int, default=None, help="Cap val images (balanced)")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--weights", type=Path, default=None, help="Score with this bundle instead of the shipped one")
    args = p.parse_args()
    print(
        collect_errors(
            k=args.k,
            max_images=args.max_images,
            use_tta=False if args.no_tta else None,
            weights=args.weights,
        )
    )
