"""Side-by-side robustness comparison of two heads, for the A/B promotion decision.

    python scripts/compare_ab.py --baseline results/robustness_table.csv \
        --candidate results/ab/robustness_table.csv --out results/ab_summary.csv

Prints a per-transform AUC delta plus the two numbers the decision rests on: does the
candidate lift the weak transforms without giving back clean accuracy? Bootstrap CIs are
carried through so a delta inside overlapping CIs is not read as a real gain.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import enable_utf8_stdout


def read(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return {r["transform"]: r for r in csv.DictReader(f)}


def main() -> None:
    enable_utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, default=Path("results/robustness_table.csv"))
    p.add_argument("--candidate", type=Path, default=Path("results/ab/robustness_table.csv"))
    p.add_argument("--out", type=Path, default=Path("results/ab_summary.csv"))
    args = p.parse_args()

    base, cand = read(args.baseline), read(args.candidate)
    names = [n for n in base if n in cand]
    if not names:
        raise SystemExit("No shared transforms between the two tables")

    rows, deltas = [], []
    for n in names:
        b, c = float(base[n]["auc"]), float(cand[n]["auc"])
        # Disjoint bootstrap CIs => the gain survives resampling noise.
        sep = float(cand[n]["auc_lo"]) > float(base[n]["auc_hi"]) or float(base[n]["auc_lo"]) > float(cand[n]["auc_hi"])
        rows.append(
            {
                "transform": n,
                "auc_baseline": f"{b:.4f}",
                "auc_candidate": f"{c:.4f}",
                "delta_auc": f"{c - b:+.4f}",
                "acc_baseline": base[n]["acc"],
                "acc_candidate": cand[n]["acc"],
                "delta_acc": f"{float(cand[n]['acc']) - float(base[n]['acc']):+.4f}",
                "ci_disjoint": "yes" if sep else "no",
            }
        )
        if n != "clean":
            deltas.append(c - b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    width = max(len(r["transform"]) for r in rows)
    for r in rows:
        flag = "  *" if r["ci_disjoint"] == "yes" else ""
        print(
            f"{r['transform']:<{width}}  base {r['auc_baseline']}  cand {r['auc_candidate']}  "
            f"delta {r['delta_auc']}{flag}"
        )
    clean_d = float(next(r["delta_auc"] for r in rows if r["transform"] == "clean")) if "clean" in base else float("nan")
    worst_b = min((float(base[n]["auc"]) for n in names if n != "clean"), default=float("nan"))
    worst_c = min((float(cand[n]["auc"]) for n in names if n != "clean"), default=float("nan"))
    print(
        f"\nclean delta {clean_d:+.4f} | mean transformed delta {sum(deltas)/len(deltas):+.4f} "
        f"| worst transform {worst_b:.4f} -> {worst_c:.4f} ({worst_c - worst_b:+.4f})"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
