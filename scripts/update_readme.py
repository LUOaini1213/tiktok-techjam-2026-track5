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
    # TPR@1%FPR is the deployment number (how much AIGC is caught if at most 1 in 100
    # authentic images may be flagged); shown whenever the table carries it.
    has_tpr = any(r.get("tpr_at_1fpr") for r in rows)
    head = "| Transform | n | Accuracy | ROC AUC | 95% CI |" + (" TPR@1%FPR |" if has_tpr else "")
    head += "\n|---|---:|---:|---:|---|" + ("---:|" if has_tpr else "")
    body = "\n".join(
        f"| `{r['transform']}` | {r['n']} | {r['acc']} | {r['auc']} | "
        f"[{r.get('auc_lo','')}, {r.get('auc_hi','')}] |"
        + (f" {r.get('tpr_at_1fpr','')} |" if has_tpr else "")
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


def ab_section(results: Path) -> str:
    rows = _rows(results / "ab_summary.csv")
    if not rows:
        return MISSING
    head = (
        "| Transform | baseline AUC | shipped AUC | delta AUC | CIs disjoint |\n"
        "|---|---:|---:|---:|---|"
    )
    body = "\n".join(
        f"| `{r['transform']}` | {r['auc_baseline']} | {r['auc_candidate']} | "
        f"{r['delta_auc']} | {r['ci_disjoint']} |"
        for r in rows
    )
    gains = [r for r in rows if r["ci_disjoint"] == "yes" and float(r["delta_auc"]) > 0]
    losses = [r for r in rows if r["ci_disjoint"] == "yes" and float(r["delta_auc"]) < 0]
    note = (
        f"\n\n{len(gains)} transform(s) improved beyond overlapping bootstrap CIs; "
        f"{len(losses)} regressed beyond them."
    )
    return f"{head}\n{body}{note}"



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


def _abl_matrix(rows: list[dict]) -> dict:
    """Mean AUC per head over the transformed rows, plus the worst row per head."""
    heads = ["clip_only_4v", "clip_only_5v", "forensic_4v", "forensic_5v"]
    per = {h: [] for h in heads}
    worst = {h: (None, 9.0) for h in heads}
    for r in rows:
        if r["transform"] == "clean":
            continue
        h, a = r["head"], float(r["auc"])
        if h in per:
            per[h].append(a)
            if a < worst[h][1]:
                worst[h] = (r["transform"], a)
    mean = {h: (sum(v) / len(v) if v else float("nan")) for h, v in per.items()}
    return {"mean": mean, "worst": worst}


def ablation_section(results: Path) -> str:
    rows = _rows(results / "ablation_table.csv")
    if not rows:
        return MISSING
    m = _abl_matrix(rows)
    clean = {r["head"]: float(r["auc"]) for r in rows if r["transform"] == "clean"}
    nan = float("nan")
    out = [
        "Mean AUC over the 14 transformed conditions (held-out SID-Set test slice, n = 1400 per cell):",
        "",
        "| | 4 views (jpeg/blur/resize) | 5 views (+ noise) |",
        "|---|---:|---:|",
        f"| **CLIP-only** | {m['mean']['clip_only_4v']:.4f} *(UnivFD-style baseline)* | {m['mean']['clip_only_5v']:.4f} |",
        f"| **CLIP + forensic** | **{m['mean']['forensic_4v']:.4f}** (down) | **{m['mean']['forensic_5v']:.4f}** *(shipped)* |",
        "",
        f"Worst single transform: baseline {m['worst']['clip_only_4v'][1]:.4f} (`{m['worst']['clip_only_4v'][0]}`), "
        f"forensic alone **{m['worst']['forensic_4v'][1]:.4f}** (`{m['worst']['forensic_4v'][0]}`), "
        f"shipped {m['worst']['forensic_5v'][1]:.4f} (`{m['worst']['forensic_5v'][0]}`). "
        f"Clean AUC: baseline {clean.get('clip_only_4v', nan):.4f}, shipped {clean.get('forensic_5v', nan):.4f}.",
    ]
    demo = _rows(results / "ablation_demo_table.csv")
    if demo:
        by: dict = {}
        for r in demo:
            by.setdefault(r["transform"], {})[r["head"]] = float(r["auc"])
        out += [
            "",
            "Cross-source (WildFake demo subset, a generator family absent from training, n = 2000 per cell):",
            "",
            "| transform | UnivFD-style | + noise view | + forensic | shipped |",
            "|---|---:|---:|---:|---:|",
        ]
        for t in ["clean", "jpeg_30", "resize_0.25", "noise_0.05"]:
            if t in by:
                b = by[t]
                out.append(
                    f"| `{t}` | {b.get('clip_only_4v', nan):.4f} | {b.get('clip_only_5v', nan):.4f} | "
                    f"{b.get('forensic_4v', nan):.4f} | **{b.get('forensic_5v', nan):.4f}** |"
                )
    if (results / "ablation_chart.png").exists():
        out += ["", "![Ablation: features x training views vs a UnivFD-style baseline](results/ablation_chart.png)"]
    return "\n".join(out)


def leakage_section(results: Path) -> str:
    path = results / "leakage" / "summary.json"
    if not path.exists():
        return MISSING
    d = json.loads(path.read_text(encoding="utf-8"))
    pr = d.get("probes", {})
    ctl = d.get("control", {})
    labels = {
        "clip_only_4v": "CLIP-only, 4 views (UnivFD-style)",
        "clip_only_5v": "CLIP-only, 5 views",
        "forensic_4v": "CLIP + forensic, 4 views",
        "forensic_5v": "CLIP + forensic, 5 views (shipped)",
    }
    out = [
        f"On the held-out test slice ({d['n']} images; both classes are stored as JPEG Q95 by our own download pipeline):",
        "",
        "**1. Does the shortcut exist in the data?** Hand-written scalars, no model. "
        "Separability is the AUROC of the scalar on its own, direction-agnostic; 0.5 means none.",
        "",
        "| probe | separability | mean real | mean fake |",
        "|---|---:|---:|---:|",
    ]
    for name, v in pr.items():
        out.append(f"| `{name}` | **{v['separability']:.4f}** | {v['mean_real']:.4f} | {v['mean_fake']:.4f} |")
    out += [
        "",
        "**2. Does the model use it?** Every image re-encoded at JPEG Q95, both classes identically, "
        "one and two extra generations (which equalises compression history), then re-scored:",
        "",
        "| head | clean AUC | +1 generation | +2 generations | delta (+1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for h, lab in labels.items():
        if h in ctl:
            c = ctl[h]
            out.append(
                f"| {lab} | {c['clean']:.4f} | {c['jpeg95_x1']:.4f} | {c['jpeg95_x2']:.4f} | {d['delta_x1'][h]:+.4f} |"
            )
    return "\n".join(out)


def by_class_section(results: Path) -> str:
    rows = _rows(results / "robustness_by_class.csv")
    if not rows:
        return MISSING
    head = (
        "| Transform | AUC: real vs **all AIGC** (headline) | AUC: real vs **fully synthetic** | "
        "AUC: real vs **tampered** | TPR@1%FPR all | TPR@1%FPR synthetic |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    body = "\n".join(
        f"| `{r['transform']}` | {r['auc_all']} | {r['auc_synthetic']} [{r['auc_synthetic_lo']}, {r['auc_synthetic_hi']}] | "
        f"{r['auc_tampered']} | {r['tpr1_all']} | {r['tpr1_synthetic']} |"
        for r in rows
    )
    syn = [float(r["auc_synthetic"]) for r in rows if r["transform"] != "clean"]
    tam = [float(r["auc_tampered"]) for r in rows if r["transform"] != "clean"]
    n = rows[0]
    note = (
        f"\n\nn per row: {n['n_all']} (all), {n['n_synthetic']} (real + fully synthetic), "
        f"{n['n_tampered']} (real + tampered). "
        + (f"Mean over transformed rows: fully synthetic {sum(syn)/len(syn):.4f}, tampered {sum(tam)/len(tam):.4f}." if syn else "")
    )
    return f"{head}\n{body}{note}"


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
        ("AB_TABLE", ab_section(results)),
        ("BY_CLASS", by_class_section(results)),
        ("ABLATION", ablation_section(results)),
        ("LEAKAGE", leakage_section(results)),
        ("ERROR_ANALYSIS", error_section(results)),
    ):
        text = replace_section(text, name, body)
    readme.write_text(text, encoding="utf-8")
    print(f"Updated {readme}")


if __name__ == "__main__":
    main()
