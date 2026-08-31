from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = ROOT
    cfg["artifacts_dir"] = ROOT / cfg.get("artifacts_dir", "artifacts")
    cfg["data_dir"] = ROOT / cfg.get("data_dir", "data/sid_subset")
    cfg["results_dir"] = ROOT / cfg.get("results_dir", "results")
    return cfg


def model_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg["artifacts_dir"] / "repostguard.joblib"
