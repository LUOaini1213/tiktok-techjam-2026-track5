#!/usr/bin/env python3
"""Render the 2x2 ablation as grouped bars with bootstrap CIs.

Reads results/ablation_table.csv (in-distribution, SID-Set held-out slice) and, when
present, results/ablation_demo_table.csv (cross-source, WildFake) as a second panel.
Writes results/ablation_chart.png.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import enable_utf8_stdout

HEADS = [
    ("clip_only_4v", "CLIP-only, 4 views  (UnivFD-style baseline)", "#8a97a8"),
    ("clip_only_5v", "CLIP-only, 5 views  (+noise)", "#c9a84b"),
    ("forensic_4v", "CLIP + forensic, 4 views", "#e0705f"),
    ("forensic_5v", "CLIP + forensic, 5 views  (shipped)", "#3b7dd8"),
]


def load(path: Path):
    if not path.exists():
        return None, []
    table = defaultdict(dict)
    order = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["transform"] not in order:
                order.append(r["transform"])
            table[r["transform"]][r["head"]] = r
    return table, order


def panel(ax, table, order, title):
    import numpy as np

    x = np.arange(len(order))
    w = 0.2
    for i, (key, label, colour) in enumerate(HEADS):
        auc = [float(table[t][key]["auc"]) for t in order]
        lo = [float(table[t][key]["auc_lo"]) for t in order]
        hi = [float(table[t][key]["auc_hi"]) for t in order]
        yerr = [[a - l for a, l in zip(auc, lo)], [h - a for a, h in zip(auc, hi)]]
        ax.bar(x + (i - 1.5) * w, auc, w, label=label, color=colour, yerr=yerr,
               capsize=2, ecolor="#444", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ROC AUC")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    enable_utf8_stdout()
    cfg = load_config()
    res = cfg["results_dir"]
    sid, sid_order = load(res / "ablation_table.csv")
    if sid is None:
        raise SystemExit("No results/ablation_table.csv -- run scripts/ablation.py first")
    demo, demo_order = load(res / "ablation_demo_table.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if demo else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4.6 * n_panels), squeeze=False)
    panel(axes[0][0], sid, sid_order,
          "In-distribution: SID-Set held-out test slice (n=1400 per bar, 95% bootstrap CI)")
    axes[0][0].set_ylim(0.78, 1.0)
    if demo:
        panel(axes[1][0], demo, demo_order,
              "Cross-source: WildFake demo subset, never trained on (n=2000 per bar)")
        axes[1][0].set_ylim(0.5, 1.0)
    axes[0][0].legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    fig.suptitle("RepostGuard ablation: features x training views, vs a UnivFD-style baseline",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    dest = res / "ablation_chart.png"
    fig.savefig(dest, dpi=130)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
