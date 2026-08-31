from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.error_analysis import collect_errors

if __name__ == "__main__":
    print(collect_errors())
