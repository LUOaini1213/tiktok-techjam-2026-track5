#!/usr/bin/env python3
"""Ablation and published-baseline comparison, over the full Track-5 transform grid.

Two axes, crossed 2x2:

    features:       CLIP-only (512-D)  vs  CLIP + native-resolution forensic (540-D)
    training views: 4 (clean/jpeg/blur/resize)  vs  5 (+ noise)

The CLIP-only / 4-view corner is a **UnivFD-style linear probe** (Ojha et al., CVPR 2023:
a linear classifier on frozen CLIP features, trained with jpeg/blur augmentation) -- i.e.
a published method, not a strawman we invented. It is the reference the shipped
configuration should have to beat.

Why this is cheap: the feature vector is `[clip | forensic]` with forensic as the trailing
28 dims, so the CLIP-only variant is a column slice of the same embedding -- no extra CLIP
forward passes. Each transformed image is embedded ONCE and scored by all four heads, so
the whole 2x2 over 15 transforms costs one sweep instead of four.

All four heads are fit and sigmoid-calibrated identically on the same slices as the shipped
model, so accuracy at 0.5 is comparable and not just AUC.

    python scripts/ablation.py --transforms clean,jpeg_30 --out results/_abl/p0.csv
    python scripts/ablation.py --merge results/_abl/p*.csv --out results/ablation_table.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.augment import EVAL_PRESETS, apply_named
from src.config import load_config
from src.data import records_from_folders
from src.eval_robustness import _load_test_image_set, balanced_val_slice, stratified_bootstrap_auc_ci
from src.io_utils import enable_utf8_stdout, open_image
from src.score import embed_for_score
from src.train_signal import fit_with_training_signal, positive_proba
from train import calibrate, group_disjoint_split, load_npz, merge_extra_views

VARIANT = "proj"          # the CV-selected variant the project ships
CLIP_DIM = 512            # proj CLIP width; forensic occupies dims 512:540
FIELDS = ["transform", "head", "n", "acc", "auc", "auc_lo", "auc_hi"]

# (key, label, feature width, extra training-view caches)
HEADS = [
    ("clip_only_4v", "CLIP-only, 4 views (UnivFD-style baseline)", CLIP_DIM, []),
    ("clip_only_5v", "CLIP-only, 5 views (+noise)", CLIP_DIM, ["noise"]),
    ("forensic_4v", "CLIP + forensic, 4 views", None, []),
    ("forensic_5v", "CLIP + forensic, 5 views (SHIPPED)", None, ["noise"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transforms", default=None, help="Comma-separated EVAL_PRESETS names")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "ablation_table.csv")
    p.add_argument("--merge", nargs="+", type=Path, default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument(
        "--source",
        choices=("sid", "demo"),
        default="sid",
        help="sid = held-out SID-Set test slice (in-distribution); "
             "demo = WildFake demonstration cache (cross-source, never trained on)",
    )
    p.add_argument("--bootstrap_reps", type=int, default=2000)
    return p.parse_args()


def merge_parts(parts, dest: Path) -> Path:
    order = {n: i for i, (n, _, _) in enumerate(EVAL_PRESETS)}
    heads = {k: i for i, (k, _, _, _) in enumerate(HEADS)}
    rows = []
    for part in parts:
        if Path(part).exists():
            with Path(part).open(newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    seen = {(r["transform"], r["head"]): r for r in rows}
    # Completeness is per transform present: every head must have scored every transform
    # that any part contains. (The cross-source run covers 4 transforms, not all 15.)
    present = sorted({t for t, _ in seen}, key=order.get)
    if not present:
        raise SystemExit("No ablation rows found in the given parts")
    missing = [(t, h) for t in present for h in heads if (t, h) not in seen]
    if missing:
        raise SystemExit(f"Merged ablation incomplete, missing {len(missing)} cells, e.g. {missing[:3]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(seen, key=lambda k: (order[k[0]], heads[k[1]])):
            w.writerow(seen[key])
    print(f"Wrote {dest} ({len(seen)} cells)")
    return dest


def build_heads(cfg):
    """Fit + calibrate all four heads from the cached features. No CLIP forwards."""
    base = load_npz(cfg["artifacts_dir"] / "features_train.npz")
    Xv, yv, _, _ = load_npz(cfg["artifacts_dir"] / "features_val.npz")
    cal_idx, test_idx = group_disjoint_split(len(yv), 0.30, cfg["seed"])

    cache = {"noise": cfg["artifacts_dir"] / "features_train_noise.npz"}
    fitted = {}
    for key, label, dim, extras in HEADS:
        Xtr, ytr, sidtr, vpi = base
        if extras:
            paths = [cache[e] for e in extras]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise SystemExit(f"{key}: missing extra-view cache {missing} -- run scripts/extract_extra_view.py")
            Xtr, ytr, sidtr, vpi = merge_extra_views(Xtr, ytr, sidtr, vpi, paths)
        d = dim or Xtr[VARIANT].shape[1]
        model = fit_with_training_signal(
            Xtr[VARIANT][:, :d], ytr, sidtr, views_per_image=vpi, tampered_weight=5.0, C=1.0
        )
        model = calibrate(model, Xv[VARIANT][cal_idx][:, :d], yv[cal_idx])
        s = positive_proba(model, Xv[VARIANT][test_idx][:, :d])
        print(f"  {label:46s} dim={d:4d} views={vpi}  cached-clean AUC={roc_auc_score(yv[test_idx], s):.4f}")
        fitted[key] = (model, d)
    return fitted


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    cfg = load_config()

    if args.merge:
        merge_parts(args.merge, args.out)
        return

    presets = EVAL_PRESETS
    if args.transforms:
        wanted = {t.strip() for t in args.transforms.split(",")}
        unknown = wanted - {n for n, _, _ in EVAL_PRESETS}
        if unknown:
            raise SystemExit(f"Unknown transform(s): {sorted(unknown)}")
        presets = [p for p in EVAL_PRESETS if p[0] in wanted]

    print("Fitting heads from cached features:")
    heads = build_heads(cfg)

    if args.source == "demo":
        # Cross-source: the WildFake demonstration cache, never used for training.
        cache = cfg["root"] / "data" / "demo_subset"
        per = (args.max_images // 2) if args.max_images else 1000
        rows = []
        for label in (0, 1):
            for path in sorted(cache.glob(f"{label}_*.png"))[:per]:
                rows.append({"path": str(path), "label": label})
        if not rows:
            raise SystemExit(f"No demo images under {cache}. Run scripts/eval_demo.py --fetch_only first.")
    else:
        rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]
        test_set = _load_test_image_set(cfg)
        if test_set is not None:
            rows = [r for r in rows if str(Path(r["path"])) in test_set]
        rows = balanced_val_slice(rows, args.max_images)
        if not rows:
            raise SystemExit("No val images. Run scripts/download_data.py first.")
    print(f"source={args.source}  n={len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name, op, param in presets:
            y = np.array([int(r["label"]) for r in rows])
            scores = {k: [] for k in heads}
            for row in tqdm(rows, desc=name):
                # ONE embedding per image, shared by every head.
                feat = embed_for_score(
                    apply_named(open_image(Path(row["path"])), op, param),
                    model_id=cfg["clip_model_id"],
                    use_forensic=True,
                    use_tta=bool(cfg.get("use_tta", True)),
                    tta_jpeg_quality=cfg["tta_jpeg_quality"],
                    tta_resize_scale=cfg["tta_resize_scale"],
                    variant=VARIANT,
                )
                for key, (model, d) in heads.items():
                    scores[key].append(float(positive_proba(model, feat[:d].reshape(1, -1))[0]))
            for key in heads:
                s = np.array(scores[key])
                acc = float(accuracy_score(y, (s >= 0.5).astype(int)))
                try:
                    auc = float(roc_auc_score(y, s))
                    lo, hi = stratified_bootstrap_auc_ci(y, s, reps=args.bootstrap_reps, seed=cfg["seed"])
                except ValueError:
                    auc = lo = hi = float("nan")
                writer.writerow({
                    "transform": name, "head": key, "n": len(y),
                    "acc": f"{acc:.4f}", "auc": f"{auc:.4f}",
                    "auc_lo": f"{lo:.4f}", "auc_hi": f"{hi:.4f}",
                })
                f.flush()
            print(f"  {name}: " + "  ".join(f"{k}={np.mean(scores[k]):.3f}" for k in heads))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
