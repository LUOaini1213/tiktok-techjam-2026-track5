from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .augment import jpeg_compress
from .config import load_config, model_path
from .data import records_from_folders
from .features import DEFAULT_FEATURE_VARIANT
from .io_utils import open_image
from .model import load_bundle
from .score import embed_for_score, score_from_embed


def collect_errors(
    k: int = 8,
    max_images: int | None = None,
    use_tta: bool | None = None,
    weights: Path | None = None,
) -> Path:
    cfg = load_config()
    bundle = load_bundle(weights or model_path(cfg))
    meta = bundle["meta"]
    variant = meta.get("feature_variant", DEFAULT_FEATURE_VARIANT)
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]
    # Exclude calibration images when a cal/test split exists (same as eval_robustness).
    from .eval_robustness import _load_test_image_set, balanced_val_slice

    test_set = _load_test_image_set(cfg)
    if test_set is not None:
        rows = [r for r in rows if str(Path(r["path"])) in test_set]
    if max_images:
        rows = balanced_val_slice(rows, max_images)
    if use_tta is None:
        use_tta = bool(meta.get("use_tta", True))

    scored = []
    for row in rows:
        im = open_image(Path(row["path"]))
        feat = embed_for_score(
            im,
            model_id=meta.get("clip_model_id", cfg["clip_model_id"]),
            use_forensic=meta.get("use_forensic", True),
            use_tta=use_tta,
            variant=variant,
        )
        pred = score_from_embed(bundle, feat)
        jpeg = jpeg_compress(im, 30)
        feat_j = embed_for_score(
            jpeg,
            model_id=meta.get("clip_model_id", cfg["clip_model_id"]),
            use_forensic=meta.get("use_forensic", True),
            use_tta=use_tta,
            variant=variant,
        )
        pred_j = score_from_embed(bundle, feat_j)
        scored.append({**row, "pred": pred, "pred_jpeg30": pred_j})

    fps = [r for r in scored if r["label"] == 0 and r["pred"] >= 0.5]
    fns = [r for r in scored if r["label"] == 1 and r["pred"] < 0.5]
    fps.sort(key=lambda r: r["pred"], reverse=True)
    fns.sort(key=lambda r: r["pred"])

    out = {
        "summary": _summarize(scored, fps, fns),
        "false_positives": fps[:k],
        "false_negatives": fns[:k],
        "note": (
            "FP: authentic images scored as AIGC (creator harm). "
            "FN: generated images missed, especially after JPEG-30."
        ),
    }
    dest = cfg["results_dir"] / "error_analysis.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def _fpr_at_tpr(y: np.ndarray, s: np.ndarray, target_tpr: float = 0.95) -> float:
    """FPR at the lowest threshold reaching `target_tpr` recall on AIGC.

    This is the creator-harm number: how many authentic images we would flag if the
    product insisted on catching 95% of generated ones.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    hit = np.where(tpr >= target_tpr)[0]
    return float(fpr[hit[0]]) if len(hit) else float("nan")


def _summarize(scored: list[dict], fps: list[dict], fns: list[dict]) -> dict:
    """Rates and score drift behind the FP/FN examples, at the shipped 0.5 threshold."""
    y = np.array([int(r["label"]) for r in scored])
    s = np.array([float(r["pred"]) for r in scored])
    sj = np.array([float(r["pred_jpeg30"]) for r in scored])
    n_real = int((y == 0).sum())
    n_fake = int((y == 1).sum())

    def _auc(scores: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y, scores))
        except ValueError:
            return float("nan")

    return {
        "n": len(scored),
        "n_real": n_real,
        "n_fake": n_fake,
        "threshold": 0.5,
        "n_false_positives": len(fps),
        "n_false_negatives": len(fns),
        "fpr": len(fps) / n_real if n_real else float("nan"),
        "fnr": len(fns) / n_fake if n_fake else float("nan"),
        "auc_clean": _auc(s),
        "auc_jpeg30": _auc(sj),
        "fpr_at_95tpr_clean": _fpr_at_tpr(y, s),
        "fpr_at_95tpr_jpeg30": _fpr_at_tpr(y, sj),
        "mean_score_drift_jpeg30_real": float(np.mean(sj[y == 0] - s[y == 0])) if n_real else float("nan"),
        "mean_score_drift_jpeg30_fake": float(np.mean(sj[y == 1] - s[y == 1])) if n_fake else float("nan"),
        "n_flips_to_fp_jpeg30": int(((y == 0) & (s < 0.5) & (sj >= 0.5)).sum()),
        "n_flips_to_fn_jpeg30": int(((y == 1) & (s >= 0.5) & (sj < 0.5)).sum()),
    }
