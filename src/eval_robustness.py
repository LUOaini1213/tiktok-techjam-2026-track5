from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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


TABLE_FIELDS = ["transform", "n", "acc", "auc", "auc_lo", "auc_hi", "tpr_at_1fpr", "tpr_at_5fpr"]


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """TPR at the largest operating point whose FPR does not exceed `target`.

    TPR@1%FPR is the deployment number for a moderation detector: how many generated
    images get caught if at most 1 in 100 authentic images may be wrongly flagged.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    return float(tpr[ok[-1]]) if len(ok) else float("nan")


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
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name in order:
            # older parts may lack the TPR columns; write them blank rather than fail
            writer.writerow({k: seen[name].get(k, "") for k in TABLE_FIELDS})
    return dest


def evaluate_robustness(
    max_images: int | None = None,
    use_tta: bool | None = None,
    table_path: Path | None = None,
    bootstrap_reps: int = 2000,
    transforms: list[str] | None = None,
    weights: Path | None = None,
    scores_dir: Path | None = None,
    crops: int = 0,
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
    suffix = f"_crops{crops}" if crops else ""
    dest = Path(table_path) if table_path else cfg["results_dir"] / f"robustness_table{suffix}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Per-image scores are persisted so any later metric (a new operating point, a
    # per-class breakdown, a paired test against another head) never needs a re-run.
    # Fixed location, NOT relative to the table path: sharded runs write their part CSVs to
    # a scratch dir, and the scores must still land in one place.
    scores_dir = Path(scores_dir) if scores_dir else cfg["results_dir"] / f"scores{suffix}"
    scores_dir.mkdir(parents=True, exist_ok=True)

    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
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
                    crops=crops,
                )
                scores.append(score_from_embed(bundle, feat))
                y_true.append(int(row["label"]))
            y = np.array(y_true)
            s = np.array(scores)
            np.savez_compressed(
                scores_dir / f"{name}.npz",
                y=y, score=s, path=np.array([Path(r["path"]).name for r in rows]),
            )
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
                    "tpr_at_1fpr": f"{tpr_at_fpr(y, s, 0.01):.4f}",
                    "tpr_at_5fpr": f"{tpr_at_fpr(y, s, 0.05):.4f}",
                }
            )
            f.flush()
    return dest
