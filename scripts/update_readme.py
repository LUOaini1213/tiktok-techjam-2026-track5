"""Populate the README's generated sections from results/.

Idempotent: each section lives between `<!-- NAME -->` and `<!-- /NAME -->` and is
rewritten in place, so re-running after a fresh eval just refreshes the numbers.

Usage:  python scripts/update_readme.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import enable_utf8_stdout

MISSING = "_(not generated yet -- run the pipeline in 'Steps to reproduce results'.)_"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _md_table(rows: list[dict]) -> str:
    head = "| Transform | n | Accuracy | ROC AUC | 95% CI |\n|---|---:|---:|---:|---|"
    body = "\n".join(
        f"| `{r['transform']}` | {r['n']} | {r['acc']} | {r['auc']} | "
        f"[{r.get('auc_lo','')}, {r.get('auc_hi','')}] |"
        for r in rows
    )
    return f"{head}\n{body}"


def robustness_section(results: Path) -> str:
    rows = _rows(results / "robustness_table.csv")
    if not rows:
        return MISSING
    by_name = {r["transform"]: r for r in rows}
    clean = float(by_name["clean"]["auc"]) if "clean" in by_name else float("nan")
    transformed = [(r["transform"], float(r["auc"])) for r in rows if r["transform"] != "clean"]
    worst_name, worst_auc = min(transformed, key=lambda t: t[1]) if transformed else ("-", float("nan"))
    mean_auc = sum(a for _, a in transformed) / len(transformed) if transformed else float("nan")

    parts = [_md_table(rows), ""]
    parts.append(
        f"**Clean AUC {clean:.4f}; mean AUC across the {len(transformed)} transformed conditions "
        f"{mean_auc:.4f} ({mean_auc - clean:+.4f} vs clean); worst condition `{worst_name}` at "
        f"{worst_auc:.4f} ({worst_auc - clean:+.4f} vs clean).** Every row is the same held-out test slice "
        f"({rows[0]['n']} images, calibration images excluded) re-scored through the shipped "
        "inference path after the transform, so clean and transformed numbers are directly comparable."
    )
    if (results / "robustness_chart.png").exists():
        parts += ["", "![Robustness: clean vs social-media transforms](results/robustness_chart.png)"]
    return "\n".join(parts)


def demo_section(results: Path) -> str:
    rows = _rows(results / "demo_benchmark.csv")
    if not rows:
        return MISSING
    clean = next((r for r in rows if r["transform"] == "clean"), None)
    out = [_md_table(rows)]
    if clean:
        out += [
            "",
            f"**Cross-source clean AUC {clean['auc']} on {clean['n']} balanced images "
            "from a generator family (DALL-E-3) absent from SID-Set training** -- an "
            "out-of-distribution check, not a tuning target.",
        ]
    return "\n".join(out)


def error_section(results: Path) -> str:
    path = results / "error_analysis.json"
    if not path.exists():
        return MISSING
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data.get("summary")
    if not s:
        return MISSING
    return "\n".join(
        [
            f"On the held-out test slice ({s['n']} images: {s['n_real']} real, {s['n_fake']} AIGC) "
            f"at the shipped 0.5 threshold:",
            "",
            f"- **False positives** (authentic flagged AIGC -- direct creator harm): "
            f"{s['n_false_positives']}/{s['n_real']} = **{s['fpr']:.2%} FPR**.",
            f"- **False negatives** (generated images missed): "
            f"{s['n_false_negatives']}/{s['n_fake']} = **{s['fnr']:.2%} FNR**.",
            f"- **FPR@95%TPR**: {s['fpr_at_95tpr_clean']:.2%} clean, "
            f"{s['fpr_at_95tpr_jpeg30']:.2%} after JPEG-30 -- the price in flagged authentic "
            "images if the product insisted on catching 95% of AIGC.",
            f"- **JPEG-30 score drift**: real {s['mean_score_drift_jpeg30_real']:+.4f}, "
            f"AIGC {s['mean_score_drift_jpeg30_fake']:+.4f} mean change in P(AIGC); "
            f"{s['n_flips_to_fp_jpeg30']} real images flip into false positives and "
            f"{s['n_flips_to_fn_jpeg30']} AIGC images flip into false negatives. "
            f"AUC {s['auc_clean']:.4f} clean vs {s['auc_jpeg30']:.4f} after JPEG-30.",
            "",
            "The `k` highest-scoring FPs and lowest-scoring FNs, with their per-image "
            "clean and JPEG-30 scores, are in `results/error_analysis.json`.",
        ]
    )


def replace_section(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"README is missing the {start} marker")
    j = text.find(end, i)
    if j < 0:
        raise SystemExit(f"README is missing the {end} marker")
    return text[: i + len(start)] + "\n" + body + "\n" + text[j:]


def main() -> None:
    enable_utf8_stdout()
    cfg = load_config()
    results = cfg["results_dir"]
    readme = cfg["root"] / "README.md"
    text = readme.read_text(encoding="utf-8")
    for name, body in (
        ("ROBUSTNESS_TABLE", robustness_section(results)),
        ("DEMO_TABLE", demo_section(results)),
        ("ERROR_ANALYSIS", error_section(results)),
    ):
        text = replace_section(text, name, body)
    readme.write_text(text, encoding="utf-8")
    print(f"Updated {readme}")


if __name__ == "__main__":
    main()
