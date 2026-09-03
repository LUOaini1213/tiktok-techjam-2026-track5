#!/usr/bin/env python3
"""Per-class robustness breakdown from the persisted per-image scores.

Our headline numbers score real vs ALL AIGC, where AIGC includes SID-Set's tampered class
(locally edited real photos). Most published numbers -- and most other Track-5 entries --
score real vs fully-synthetic only, an easier definition. To compare like with like this
writes both, plus the tampered-only view, for every transform, from results/scores/*.npz.
No model runs: everything is derived from scores already on disk.

    python scripts/per_class_table.py        # -> results/robustness_by_class.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import EVAL_PRESETS
from src.config import load_config
from src.eval_robustness import stratified_bootstrap_auc_ci, tpr_at_fpr
from src.io_utils import enable_utf8_stdout
from src.train_signal import sid_label_from_path

FIELDS = [
    "transform", "n_all", "n_synthetic", "n_tampered",
    "auc_all", "auc_synthetic", "auc_synthetic_lo", "auc_synthetic_hi", "auc_tampered",
    "tpr1_all", "tpr1_synthetic", "tpr5_all", "tpr5_synthetic",
]


def main() -> None:
    enable_utf8_stdout()
    cfg = load_config()
    scores_dir = cfg["results_dir"] / "scores"
    rows_out = []
    for name, _, _ in EVAL_PRESETS:
        path = scores_dir / f"{name}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        y, s = d["y"], d["score"]
        sid = np.array([sid_label_from_path(str(p)) for p in d["path"]])
        syn = sid != 2   # real + fully synthetic
        tam = sid != 1   # real + tampered
        lo, hi = stratified_bootstrap_auc_ci(y[syn], s[syn], reps=2000, seed=cfg["seed"])
        rows_out.append({
            "transform": name,
            "n_all": int(len(y)), "n_synthetic": int(syn.sum()), "n_tampered": int(tam.sum()),
            "auc_all": f"{roc_auc_score(y, s):.4f}",
            "auc_synthetic": f"{roc_auc_score(y[syn], s[syn]):.4f}",
            "auc_synthetic_lo": f"{lo:.4f}", "auc_synthetic_hi": f"{hi:.4f}",
            "auc_tampered": f"{roc_auc_score(y[tam], s[tam]):.4f}",
            "tpr1_all": f"{tpr_at_fpr(y, s, 0.01):.4f}",
            "tpr1_synthetic": f"{tpr_at_fpr(y[syn], s[syn], 0.01):.4f}",
            "tpr5_all": f"{tpr_at_fpr(y, s, 0.05):.4f}",
            "tpr5_synthetic": f"{tpr_at_fpr(y[syn], s[syn], 0.05):.4f}",
        })
    if not rows_out:
        raise SystemExit(f"No per-image scores under {scores_dir}; run scripts/make_tables.py first")
    dest = cfg["results_dir"] / "robustness_by_class.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    for r in rows_out:
        print(f"{r['transform']:12s} all={r['auc_all']}  synthetic={r['auc_synthetic']}  tampered={r['auc_tampered']}  "
              f"TPR@1% all={r['tpr1_all']} synthetic={r['tpr1_synthetic']}")
    print(f"Wrote {dest} ({len(rows_out)} transforms)")


if __name__ == "__main__":
    main()
