from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from .augment import EVAL_PRESETS, apply_named
from .config import load_config, model_path
from .data import records_from_folders
from .features import DEFAULT_FEATURE_VARIANT
from .io_utils import open_image
from .model import load_bundle
from .score import embed_for_score, score_from_embed


def balanced_val_slice(rows: list[dict], max_images: int | None) -> list[dict]:
    if not max_images:
        return list(rows)
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    n0 = max(1, max_images // 2)
    n1 = max(1, max_images - n0)
    return by_label.get(0, [])[:n0] + by_label.get(1, [])[:n1]


def _load_test_image_set(cfg) -> set[str] | None:
    """If train.py wrote a held-out test slice, restrict evaluation to it (exclude cal)."""
    path = cfg["artifacts_dir"] / "test_images.json"
    if not path.exists():
        return None
    try:
        return {str(Path(p)) for p in json.loads(path.read_text(encoding="utf-8"))}
    except Exception:
        return None


def stratified_bootstrap_auc_ci(
    y: np.ndarray, s: np.ndarray, reps: int = 2000, seed: int = 2026
) -> tuple[float, float]:
    """95% percentile CI for AUC via per-class resampling with replacement.

    Point estimate comes from the original sample (computed by the caller); here we only
    return the 2.5/97.5 percentiles of the bootstrap distribution. Each class's rows are
    resampled independently so the class balance is preserved (val has 1 view/image, so
    no cluster/group bootstrap is needed).
    """
    y = np.asarray(y)
    s = np.asarray(s)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(reps):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        bi = np.concatenate([b0, b1])
        yb, sb = y[bi], s[bi]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def merge_tables(parts: list[Path], dest: Path) -> Path:
    """Merge sharded robustness CSVs back into one table in EVAL_PRESETS order.

    Sharding exists only to use more than one CPU core: every row is produced by the
    identical code path, so a merged table is byte-identical to a single-process run.
    """
    order = [name for name, _, _ in EVAL_PRESETS]
    seen: dict[str, dict] = {}
    for part in parts:
        if not Path(part).exists():
            continue
        with Path(part).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen[row["transform"]] = row
    missing = [n for n in order if n not in seen]
    if missing:
        raise RuntimeError(f"Merged table is incomplete, missing transforms: {missing}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["transform", "n", "acc", "auc", "auc_lo", "auc_hi"])
        writer.writeheader()
        for name in order:
            writer.writerow(seen[name])
    return dest


def evaluate_robustness(
    max_images: int | None = None,
    use_tta: bool | None = None,
    table_path: Path | None = None,
    bootstrap_reps: int = 2000,
    transforms: list[str] | None = None,
    weights: Path | None = None,
) -> Path:
    # Pure argument validation, before anything expensive or destructive: a typo should
    # fail instantly rather than after a CLIP load, and must never reach the "w" open
    # below that would truncate an existing table.
    presets = EVAL_PRESETS
    if transforms:
        wanted = set(transforms)
        unknown = wanted - {n for n, _, _ in EVAL_PRESETS}
        if unknown:
            raise ValueError(f"Unknown transform(s): {sorted(unknown)}")
        presets = [p for p in EVAL_PRESETS if p[0] in wanted]

    cfg = load_config()
    bundle = load_bundle(weights or model_path(cfg))
    meta = bundle["meta"]
    variant = meta.get("feature_variant", DEFAULT_FEATURE_VARIANT)
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]

    # If a calibration/test split exists, evaluate ONLY on the held-out test images so the
    # reported robustness is not optimistic from calibration leakage.
    test_set = _load_test_image_set(cfg)
    if test_set is not None:
        rows = [r for r in rows if str(Path(r["path"])) in test_set]

    rows = balanced_val_slice(rows, max_images)
    if not rows:
        raise RuntimeError("No val images. Run scripts/download_data.py first.")

    if use_tta is None:
        use_tta = bool(meta.get("use_tta", cfg["use_tta"]))

    # Opened with "w" below, which truncates. Everything that can reject the call --
    # bad transform names, a missing model, an empty slice -- has to happen first, or a
    # failed run destroys the previous table.
    dest = Path(table_path) if table_path else cfg["results_dir"] / "robustness_table.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["transform", "n", "acc", "auc", "auc_lo", "auc_hi"]
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, op, param in presets:
            y_true = []
            scores = []
            for row in tqdm(rows, desc=name):
                im = apply_named(open_image(Path(row["path"])), op, param)
                feat = embed_for_score(
                    im,
                    model_id=meta.get("clip_model_id", cfg["clip_model_id"]),
                    use_forensic=meta.get("use_forensic", True),
                    use_tta=use_tta,
                    tta_jpeg_quality=cfg["tta_jpeg_quality"],
                    tta_resize_scale=cfg["tta_resize_scale"],
                    variant=variant,
                )
                scores.append(score_from_embed(bundle, feat))
                y_true.append(int(row["label"]))
            y = np.array(y_true)
            s = np.array(scores)
            hard = (s >= 0.5).astype(int)
            acc = float(accuracy_score(y, hard))
            try:
                auc = float(roc_auc_score(y, s))
                lo, hi = stratified_bootstrap_auc_ci(y, s, reps=bootstrap_reps, seed=cfg["seed"])
            except ValueError:
                auc = lo = hi = float("nan")
            writer.writerow(
                {
                    "transform": name,
                    "n": len(y),
                    "acc": f"{acc:.4f}",
                    "auc": f"{auc:.4f}",
                    "auc_lo": f"{lo:.4f}",
                    "auc_hi": f"{hi:.4f}",
                }
            )
            f.flush()
    return dest
