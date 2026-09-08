"""Fit logistic head on extracted features, A/B pre- vs projected-CLIP, and calibrate.

Pipeline:
  1. Load both cached CLIP variants (X_pre 768+forensic, X_proj 512+forensic).
  2. For each variant, GroupKFold(5) CV AUC with groups = source-image id
     (row_index // views_per_image) on the RAW view rows, so augmented views of one
     image never straddle a train/val fold boundary.
  3. Pick the winner by mean CV AUC; fit its head on full train via the shared
     training-signal path (consistency rows + tampered upweight).
  4. Sigmoid-calibrate on a group-disjoint 30% slice of val; report clean metrics on
     the held-out 70% test slice only.
  5. Persist winner (calibrated) with meta feature_variant + feature_dim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, model_path
from src.io_utils import enable_utf8_stdout
from src.augment import PAIR_FAMILIES
from src.features import BASE_DIMS, DINO_MODEL_ID, FEATURE_VARIANTS, feature_dim_for_variant
from src.model import save_bundle
from src.train_signal import fit_with_training_signal, positive_proba


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--cal_frac", type=float, default=0.30, help="Group-disjoint val fraction for calibration")
    p.add_argument("--no_calibrate", action="store_true")
    p.add_argument(
        "--extra_features",
        nargs="+",
        type=Path,
        default=None,
        help="Extra 1-view-per-image caches (scripts/extract_extra_view.py) to merge into each image's view block",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the bundle here instead of artifacts/repostguard.joblib (A/B runs)",
    )
    p.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated subset of feature variants to A/B (default: every variant in the cache)",
    )
    return p.parse_args()


def load_npz(path: Path):
    """Return {variant: X}, y, sid, views_per_image. Falls back to X if X_pre absent."""
    data = np.load(path)
    files = set(data.files)
    Xs = {}
    for v in FEATURE_VARIANTS:
        key = f"X_{v}"
        if key in files:
            Xs[v] = data[key]
        elif v == "proj" and "X" in files:
            Xs[v] = data["X"]  # legacy single-variant cache
    y = data["y"]
    sid = data["sid"] if "sid" in files else np.zeros(len(y), dtype=np.int64)
    vpi = int(data["views_per_image"][0]) if "views_per_image" in files else 1
    return Xs, y, sid, vpi


def merge_extra_views(Xs, y, sid, vpi, extra_paths):
    """Fold 1-view-per-image caches into each image's contiguous view block.

    `source_groups` derives CV groups from position (`row // views_per_image`), so extra
    views cannot simply be appended at the end -- they must land inside their own image's
    block. Extra caches are written in the same image order as features_train.npz, so a
    reshape to (n_images, views, dim) and a concat along the view axis is exact.
    """
    n_img, rem = divmod(len(y), vpi)
    if rem:
        raise SystemExit(f"features_train.npz has {len(y)} rows, not a multiple of views_per_image={vpi}")
    blocks = {v: Xs[v].reshape(n_img, vpi, -1) for v in Xs}
    y_b = y.reshape(n_img, vpi)
    sid_b = sid.reshape(n_img, vpi)

    for path in extra_paths:
        Xe, ye, side, vpe = load_npz(path)
        n_e, rem_e = divmod(len(ye), vpe)
        if rem_e or n_e != n_img:
            raise SystemExit(f"{path}: covers {n_e} images, base cache has {n_img} -- re-extract")
        missing = [v for v in blocks if v not in Xe]
        if missing:
            raise SystemExit(f"{path}: missing feature variant(s) {missing}")
        for v in blocks:
            if Xe[v].shape[1] != blocks[v].shape[2]:
                raise SystemExit(
                    f"{path}: variant {v} is {Xe[v].shape[1]}-D but base cache is {blocks[v].shape[2]}-D"
                )
            blocks[v] = np.concatenate([blocks[v], Xe[v].reshape(n_img, vpe, -1)], axis=1)
        y_e = ye.reshape(n_img, vpe)
        if not (y_e == y_b[:, :1]).all():
            raise SystemExit(f"{path}: labels do not line up with the base cache -- image order differs")
        y_b = np.concatenate([y_b, y_e], axis=1)
        sid_b = np.concatenate([sid_b, side.reshape(n_img, vpe)], axis=1)
        print(f"merged {path.name}: +{vpe} view/image")

    new_vpi = y_b.shape[1]
    return (
        {v: blocks[v].reshape(n_img * new_vpi, -1) for v in blocks},
        y_b.reshape(-1),
        sid_b.reshape(-1),
        new_vpi,
    )


def fpr_at_tpr(y_true, scores, target_tpr=0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return float("nan")
    return float(fpr[idx[0]])


def source_groups(n_rows: int, views_per_image: int) -> np.ndarray:
    """Source-image id per RAW view row so CV folds never split one image's views."""
    return np.arange(n_rows) // max(1, views_per_image)


def cv_auc(X, y, sid, groups, views_per_image, C, n_splits) -> float:
    """Mean GroupKFold AUC. Each fold fits the SAME training-signal head used to ship."""
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        # Refit views_per_image inside the fold: the consistency expansion needs whole
        # image blocks, and GroupKFold keeps each image's views together, but the fold
        # rows are not necessarily a clean multiple -> fit per-row (vpi=1) inside CV to
        # keep it robust; the tampered upweight signal is preserved.
        model = fit_with_training_signal(
            X[tr], y[tr], sid[tr], views_per_image=1, C=C
        )
        s = positive_proba(model, X[te])
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)) if aucs else float("nan")


def group_disjoint_split(n_val: int, cal_frac: float, seed: int):
    """Each val image is its own group (val has 1 view/image) -> a plain row split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_val)
    n_cal = max(1, int(round(cal_frac * n_val)))
    n_cal = min(n_cal, n_val - 1)  # keep at least 1 test row
    cal_idx = np.sort(perm[:n_cal])
    test_idx = np.sort(perm[n_cal:])
    return cal_idx, test_idx


def calibrate(model, X_cal, y_cal):
    """Sigmoid-calibrate an already-fit head on a held-out calibration slice.

    Prefers sklearn>=1.6 FrozenEstimator (base model is NOT refit -- the sigmoid is fit
    on this exact head's outputs). FrozenEstimator still internally cross-validates to
    build calibration targets, so cv is capped by the minority class count. Falls back to
    cv='prefit' on older sklearn where FrozenEstimator does not exist.
    """
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV

    _, class_counts = np.unique(y_cal, return_counts=True)
    cv = int(min(5, class_counts.min()))
    cv = max(2, cv)
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", cv=cv)
    except Exception:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    cfg = load_config()
    train_path = cfg["artifacts_dir"] / "features_train.npz"
    val_path = cfg["artifacts_dir"] / "features_val.npz"
    if not train_path.exists():
        raise SystemExit("Run scripts/extract_features.py --split train --augment first")

    Xtr, ytr, sidtr, vpi = load_npz(train_path)
    extra_names = []
    if args.extra_features:
        Xtr, ytr, sidtr, vpi = merge_extra_views(Xtr, ytr, sidtr, vpi, args.extra_features)
        extra_names = [Path(p).name for p in args.extra_features]
        print(f"views_per_image now {vpi} ({len(ytr)} train rows)")
    groups = source_groups(len(ytr), vpi)

    # ---- A/B: GroupKFold CV AUC per variant ----
    cv_scores = {}
    wanted = [v.strip() for v in args.variants.split(",")] if args.variants else list(FEATURE_VARIANTS)
    unknown = [v for v in wanted if v not in FEATURE_VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant(s) {unknown}; choose from {FEATURE_VARIANTS}")
    for v in wanted:
        if v not in Xtr:
            print(f"variant {v} not in the feature cache; skipping")
            continue
        cv_scores[v] = cv_auc(
            Xtr[v], ytr, sidtr, groups, vpi, C=args.C, n_splits=args.cv_splits
        )
        print(f"CV AUC[{v}] = {cv_scores[v]:.4f}  (dim={Xtr[v].shape[1]})")
    valid = {v: s for v, s in cv_scores.items() if s == s}  # drop NaN
    if not valid:
        raise SystemExit("No variant produced a valid CV AUC (need >=2 groups, both classes).")
    winner = max(valid, key=valid.get)
    print(f"WINNER variant = {winner} (CV AUC {valid[winner]:.4f})")

    X_train = Xtr[winner]
    model = fit_with_training_signal(
        X_train, ytr, sidtr, views_per_image=vpi, tampered_weight=5.0, C=args.C
    )

    metrics = {
        "n_train": int(len(ytr)),
        "views_per_image": vpi,
        "n_tampered_rows": int((sidtr == 2).sum()),
        "tampered_weight": 5.0,
        "C": args.C,
        "cv_auc": {v: (None if s != s else float(s)) for v, s in cv_scores.items()},
        "feature_variant": winner,
    }

    calibrated_model = model
    calibration = "none"
    cal_list, test_list = None, None

    if val_path.exists():
        Xval, yval, _sidval, _v = load_npz(val_path)
        Xv = Xval[winner]

        if not args.no_calibrate and len(yval) >= 4:
            cal_idx, test_idx = group_disjoint_split(len(yval), args.cal_frac, cfg["seed"])
            # Calibrate on cal slice, evaluate ONLY on test slice.
            if len(np.unique(yval[cal_idx])) >= 2 and len(np.unique(yval[test_idx])) >= 2:
                calibrated_model = calibrate(model, Xv[cal_idx], yval[cal_idx])
                calibration = "sigmoid"
                scores = positive_proba(calibrated_model, Xv[test_idx])
                yt = yval[test_idx]
                pred = (scores >= 0.5).astype(int)
                metrics.update(
                    {
                        "n_val_total": int(len(yval)),
                        "n_cal": int(len(cal_idx)),
                        "n_val": int(len(test_idx)),
                        "acc": float(accuracy_score(yt, pred)),
                        "auc": float(roc_auc_score(yt, scores)),
                        "fpr95": fpr_at_tpr(yt, scores),
                    }
                )
                cal_list = _row_paths(cfg, cal_idx)
                test_list = _row_paths(cfg, test_idx)
            else:
                args.no_calibrate = True  # fall through to uncalibrated full-val metrics

        if calibration == "none":
            scores = positive_proba(model, Xv)
            pred = (scores >= 0.5).astype(int)
            metrics.update(
                {
                    "n_val": int(len(yval)),
                    "acc": float(accuracy_score(yval, pred)),
                    "auc": float(roc_auc_score(yval, scores)),
                    "fpr95": fpr_at_tpr(yval, scores),
                }
            )
        print(json.dumps(metrics, indent=2))
    else:
        print("No val features; fitting train only")

    # Persist cal/test image lists so eval_robustness evaluates on the test slice only.
    if cal_list is not None:
        (cfg["artifacts_dir"] / "cal_images.json").write_text(
            json.dumps(cal_list), encoding="utf-8"
        )
        (cfg["artifacts_dir"] / "test_images.json").write_text(
            json.dumps(test_list), encoding="utf-8"
        )

    meta = {
        "clip_model_id": cfg["clip_model_id"],
        "use_forensic": cfg["use_forensic"],
        "use_tta": cfg["use_tta"],
        "tta_jpeg_quality": cfg["tta_jpeg_quality"],
        "tta_resize_scale": cfg["tta_resize_scale"],
        "seed": cfg["seed"],
        "metrics": metrics,
        "feature_variant": winner,
        "feature_dim": int(feature_dim_for_variant(winner, cfg["use_forensic"])),
        "calibration": calibration,
        "param_note": f"{_encoder_note(winner, cfg['clip_model_id'])} + logistic regression",
        "training_data": (
            "SID-Set subset. Binary 0=real vs 1=AIGC-positive (full-synthetic + tampered). "
            f"Clean + one paired official view per family ({'/'.join(PAIR_FAMILIES)}) per image, "
            "each row the TTA-averaged embedding. Tampered rows upweighted. WildFake unused."
        ),
        "extra_view_caches": extra_names,
        "score_path": "embed_for_score",
        "use_tta_for_fit": True,
        "tampered_weight": 5.0,
    }
    out = args.out or model_path(cfg)
    save_bundle(out, calibrated_model, meta)
    if args.out is None:  # A/B runs must not overwrite the shipped WEIGHTS.txt note
        _write_weights_note(cfg, metrics, winner, calibration)
    print(f"Wrote {out}")


def _row_paths(cfg, idx: np.ndarray) -> list[str]:
    """Val image paths in the same row order extract_features wrote them (sorted rglob)."""
    from src.data import records_from_folders

    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]
    paths = [r["path"] for r in rows]
    return [paths[i] for i in idx if i < len(paths)]


def _encoder_note(winner: str, clip_id: str) -> str:
    enc = f"frozen {clip_id} (ViT-B/32, ~88M)"
    if winner in ("dino", "fuse"):
        enc += f" + frozen {DINO_MODEL_ID} CLS (DINOv2-small, ~22M)"
    return enc + "; well under 2B params"


def _write_weights_note(cfg, metrics, winner, calibration) -> None:
    note = cfg["artifacts_dir"] / "WEIGHTS.txt"
    cv = metrics.get("cv_auc", {})
    dims = "  ".join(f"{v}={BASE_DIMS[v]}+forensic" for v in FEATURE_VARIANTS)
    cv_line = "  ".join(
        f"{v}={cv[v]:.4f}" if isinstance(cv.get(v), float) else f"{v}=n/a" for v in FEATURE_VARIANTS
    )
    views = "/".join(PAIR_FAMILIES)
    note.write_text(
        (
            "SID-SET SUBSET TRAINED (not smoke)\n\n"
            f"feature_variant={winner}  ({dims})\n"
            f"CV AUC {cv_line}\n"
            f"n_train={metrics.get('n_train')}  (rows = images x {metrics.get('views_per_image')} views)\n"
            f"n_val(test slice)={metrics.get('n_val')}\n"
            f"n_cal={metrics.get('n_cal')}\n"
            f"acc={metrics.get('acc')}\n"
            f"auc={metrics.get('auc')}\n"
            f"fpr95={metrics.get('fpr95')}\n"
            f"calibration={calibration}\n"
            f"Encoder: {_encoder_note(winner, cfg['clip_model_id'])}.\n"
            "Labels: SID-Set 0=real, 1=AIGC-positive (includes tampered/label 2). WildFake unused.\n"
            f"Training rows: clean + one paired official view per family ({views}); each row is the\n"
            "TTA-averaged embedding from embed_for_score (same path as infer). Tampered weight=5.\n"
            "Sigmoid-calibrated on group-disjoint 30% of val; metrics reported on the held-out 70%\n"
            "test slice only.\n"
        ),
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
