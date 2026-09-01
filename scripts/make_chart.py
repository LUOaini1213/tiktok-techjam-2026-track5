"""Render a compact robustness bar chart (AUC with bootstrap CI) from the CSV.

Reads results/robustness_table.csv and writes results/robustness_chart.png for the
README's Robustness Evaluation Summary. Skips gracefully if the table is missing.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import enable_utf8_stdout


def main() -> None:
    enable_utf8_stdout()
    cfg = load_config()
    src = cfg["results_dir"] / "robustness_table.csv"
    if not src.exists():
        raise SystemExit(f"No {src}; run scripts/make_tables.py first")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, aucs, los, his = [], [], [], []
    with src.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["transform"])
            aucs.append(float(row["auc"]))
            los.append(float(row.get("auc_lo", row["auc"])))
            his.append(float(row.get("auc_hi", row["auc"])))

    yerr = [
        [a - lo for a, lo in zip(aucs, los)],
        [hi - a for a, hi in zip(aucs, his)],
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(names))
    ax.bar(x, aucs, color="#3b7dd8", yerr=yerr, capsize=3, ecolor="#333")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_title("RepostGuard robustness: clean vs social-media transforms (95% bootstrap CI)")
    fig.tight_layout()

    dest = cfg["results_dir"] / "robustness_chart.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
