"""Build the clean vs transformed robustness table required by Track 5.

Single-process (the reproducible default):

    python scripts/make_tables.py

Sharded across CPU cores (identical rows, just faster wall clock): run one process per
subset of transforms writing its own part file, then merge them in preset order:

    python scripts/make_tables.py --transforms clean,jpeg_90,jpeg_70 --out results/_part0.csv
    ...
    python scripts/make_tables.py --merge results/_part*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import enable_utf8_stdout
from src.eval_robustness import evaluate_robustness, merge_tables


def main() -> None:
    enable_utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--max_images", type=int, default=None, help="Cap val images (balanced across labels)")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument(
        "--transforms",
        type=str,
        default=None,
        help="Comma-separated EVAL_PRESETS names to score (default: all)",
    )
    p.add_argument("--out", type=Path, default=None, help="Write this CSV instead of results/robustness_table.csv")
    p.add_argument("--weights", type=Path, default=None, help="Score with this bundle instead of the shipped one (A/B)")
    p.add_argument("--crops", type=int, default=0, help="Eval-time multi-crop TTA: add N corner/centre crops per image")
    p.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        help="Merge these part CSVs into results/robustness_table.csv and exit",
    )
    args = p.parse_args()

    if args.merge:
        dest = args.out or (load_config()["results_dir"] / "robustness_table.csv")
        print(f"Wrote {merge_tables(args.merge, dest)}")
        return

    path = evaluate_robustness(
        max_images=args.max_images,
        use_tta=False if args.no_tta else None,
        table_path=args.out,
        weights=args.weights,
        crops=args.crops,
        transforms=[t.strip() for t in args.transforms.split(",")] if args.transforms else None,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
