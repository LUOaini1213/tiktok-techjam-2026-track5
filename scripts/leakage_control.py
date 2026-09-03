#!/usr/bin/env python3
"""JPEG-history leakage control: is the detector reading compression history?

The concern, raised independently by several other Track-5 teams: in SID-Set the authentic
images ship as JPEG and the synthetic ones as PNG, so a detector can score well by reading
*compression history* rather than generation artefacts. Our download pipeline already
re-encodes every image to JPEG Q95, which erases the file container -- but not the history:
reals are then double-compressed and fakes single-compressed, and our forensic branch
contains block-DCT statistics aligned to the JPEG grid, exactly the feature that would
notice.

Two tests, both on the held-out test slice:

1. Shortcut-existence probes (no model). Hand-written scalars that only measure JPEG
   history or smoothness: 8x8-grid blockiness, and mean |Laplacian| (the "fakes are
   smoother" shortcut). AUROC of each scalar on its own. Near 0.5 means the data does not
   carry that shortcut; high means it does, and test 2 decides whether the model uses it.

2. Model-dependence control. Re-encode every image at JPEG Q95 -- both classes identically,
   from decoded pixels -- one and two extra generations, and re-score all four ablation
   heads (CLIP-only and CLIP+forensic, with and without the noise view). A detector that
   relies on compression history loses AUC when the history is equalised; one that reads
   generation artefacts does not. The CLIP-only heads are the control group for the
   forensic ones.

Per-image scores are persisted to results/leakage/ so any further metric is free.

    python scripts/leakage_control.py            # ~2800 embeddings, one worker
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation import HEADS, VARIANT, build_heads
from src.config import load_config
from src.data import records_from_folders
from src.eval_robustness import _load_test_image_set, balanced_val_slice, stratified_bootstrap_auc_ci
from src.io_utils import enable_utf8_stdout, open_image
from src.score import embed_for_score
from src.train_signal import positive_proba

CONDITIONS = [("clean", 0), ("jpeg95_x1", 1), ("jpeg95_x2", 2)]
FIELDS = ["condition", "head", "n", "auc", "auc_lo", "auc_hi", "tpr_at_1fpr", "tpr_at_5fpr"]


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else float("nan")


def reencode(img: Image.Image, generations: int, quality: int = 95) -> Image.Image:
    for _ in range(generations):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=False)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


# ----------------------------------------------------------------- shortcut probes
def _gray_center(img: Image.Image, size: int = 256) -> np.ndarray:
    """Native-scale centre crop, like the forensic branch -- no resize, so grids survive."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    s = min(size, h, w)
    top, left = (h - s) // 2, (w - s) // 2
    # snap to the 8-px JPEG grid so boundary positions are meaningful
    top -= top % 8
    left -= left % 8
    return g[top : top + s, left : left + s]


def blockiness(img: Image.Image) -> float:
    """Mean |diff| across 8-px block boundaries minus elsewhere (Fan & de Queiroz style).

    Positive means visible JPEG block structure. Zero-ish means none or fully smoothed.
    """
    g = _gray_center(img)
    dh = np.abs(np.diff(g, axis=1))  # horizontal neighbour differences, column x vs x+1
    dv = np.abs(np.diff(g, axis=0))
    cols = np.arange(dh.shape[1])
    rows = np.arange(dv.shape[0])
    bh = (cols + 1) % 8 == 0
    bv = (rows + 1) % 8 == 0
    h_term = dh[:, bh].mean() - dh[:, ~bh].mean()
    v_term = dv[bv, :].mean() - dv[~bv, :].mean()
    return float(h_term + v_term)


def laplacian_energy(img: Image.Image) -> float:
    """Mean |Laplacian| on the native crop: the 'generated images are smoother' shortcut."""
    g = _gray_center(img)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(np.abs(lap).mean())


PROBES = [("blockiness_8px", blockiness), ("laplacian_energy", laplacian_energy)]


def main() -> None:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--bootstrap_reps", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config()

    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]
    test_set = _load_test_image_set(cfg)
    if test_set is not None:
        rows = [r for r in rows if str(Path(r["path"])) in test_set]
    rows = balanced_val_slice(rows, args.max_images)
    if not rows:
        raise SystemExit("No val images. Run scripts/download_data.py first.")
    y = np.array([int(r["label"]) for r in rows])
    out_dir = cfg["results_dir"] / "leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}

    # ---- 1. shortcut-existence probes (no model) ----
    print("Shortcut probes on the images as stored (both classes JPEG Q95 on disk):")
    probe_vals = {name: [] for name, _ in PROBES}
    for r in tqdm(rows, desc="probes"):
        im = open_image(Path(r["path"]))
        for name, fn in PROBES:
            probe_vals[name].append(fn(im))
    summary["probes"] = {}
    for name, _ in PROBES:
        v = np.array(probe_vals[name])
        auc = roc_auc_score(y, v)
        sep = max(auc, 1 - auc)  # direction-agnostic separability
        summary["probes"][name] = {
            "auroc_raw": float(auc),
            "separability": float(sep),
            "mean_real": float(v[y == 0].mean()),
            "mean_fake": float(v[y == 1].mean()),
        }
        print(f"  {name:18s} separability={sep:.4f}  mean real={v[y==0].mean():.4f}  fake={v[y==1].mean():.4f}")
    np.savez_compressed(out_dir / "probes.npz", y=y, **{k: np.array(v) for k, v in probe_vals.items()})

    # ---- 2. model-dependence control ----
    print("Fitting the four ablation heads from cached features:")
    heads = build_heads(cfg)
    table = []
    with (out_dir / "leakage_control.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for cond, gens in CONDITIONS:
            scores = {k: [] for k in heads}
            for r in tqdm(rows, desc=cond):
                im = reencode(open_image(Path(r["path"])), gens)
                feat = embed_for_score(
                    im,
                    model_id=cfg["clip_model_id"],
                    use_forensic=True,
                    use_tta=bool(cfg.get("use_tta", True)),
                    tta_jpeg_quality=cfg["tta_jpeg_quality"],
                    tta_resize_scale=cfg["tta_resize_scale"],
                    variant=VARIANT,
                )
                for key, (model, d) in heads.items():
                    scores[key].append(float(positive_proba(model, feat[:d].reshape(1, -1))[0]))
            np.savez_compressed(out_dir / f"scores_{cond}.npz", y=y, **{k: np.array(v) for k, v in scores.items()})
            for key in heads:
                s = np.array(scores[key])
                auc = float(roc_auc_score(y, s))
                lo, hi = stratified_bootstrap_auc_ci(y, s, reps=args.bootstrap_reps, seed=cfg["seed"])
                row = {
                    "condition": cond, "head": key, "n": int(len(y)),
                    "auc": f"{auc:.4f}", "auc_lo": f"{lo:.4f}", "auc_hi": f"{hi:.4f}",
                    "tpr_at_1fpr": f"{tpr_at_fpr(y, s, 0.01):.4f}",
                    "tpr_at_5fpr": f"{tpr_at_fpr(y, s, 0.05):.4f}",
                }
                w.writerow(row)
                f.flush()
                table.append(row)
            print(f"  {cond}: " + "  ".join(f"{k}={float(r['auc']):.4f}" for k in heads for r in table if r['condition'] == cond and r['head'] == k))

    # ---- verdict ----
    by = {(r["condition"], r["head"]): float(r["auc"]) for r in table}
    summary["control"] = {
        h: {c: by[(c, h)] for c, _ in CONDITIONS} for h, *_ in HEADS
    }
    summary["delta_x1"] = {h: by[("jpeg95_x1", h)] - by[("clean", h)] for h, *_ in HEADS}
    summary["delta_x2"] = {h: by[("jpeg95_x2", h)] - by[("clean", h)] for h, *_ in HEADS}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nAUC delta after equalising compression history (both classes re-encoded Q95):")
    for h, label, *_ in HEADS:
        print(f"  {label:46s} x1 {summary['delta_x1'][h]:+.4f}   x2 {summary['delta_x2'][h]:+.4f}")
    print(f"Wrote {out_dir / 'leakage_control.csv'} and summary.json")


if __name__ == "__main__":
    main()
