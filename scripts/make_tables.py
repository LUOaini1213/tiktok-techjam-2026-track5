"""Build the clean vs transformed robustness table required by Track 5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_robustness import evaluate_robustness


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_images", type=int, default=None, help="Cap val images (balanced across labels)")
    p.add_argument("--no_tta", action="store_true")
    args = p.parse_args()
    path = evaluate_robustness(
        max_images=args.max_images,
        use_tta=False if args.no_tta else None,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
