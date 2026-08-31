"""Fit logistic regression on extracted features and print clean val metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, model_path
from src.model import save_bundle
from src.train_signal import expand_with_consistency, fit_with_training_signal, positive_proba


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def load_npz(path: Path):
    data = np.load(path)
    sid = data["sid"] if "sid" in data.files else np.zeros(len(data["y"]), dtype=np.int64)
    vpi = int(data["views_per_image"][0]) if "views_per_image" in data.files else 1
    return data["X"], data["y"], sid, vpi


def fpr_at_tpr(y_true, scores, target_tpr=0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return float("nan")
    return float(fpr[idx[0]])


def main() -> None:
    args = parse_args()
    cfg = load_config()
    train_path = cfg["artifacts_dir"] / "features_train.npz"
    val_path = cfg["artifacts_dir"] / "features_val.npz"
    if not train_path.exists():
        raise SystemExit("Run scripts/extract_features.py --split train --augment first")

    X_train, y_train, sid_train, vpi = load_npz(train_path)
    X_fit, y_fit, sid_fit = expand_with_consistency(X_train, y_train, sid_train, vpi)
    model = fit_with_training_signal(
        X_train,
        y_train,
        sid_train,
        views_per_image=vpi,
        tampered_weight=5.0,
        C=args.C,
    )

    metrics = {
        "n_train": int(len(y_fit)),
        "n_raw_views": int(len(y_train)),
        "views_per_image": vpi,
        "n_tampered_rows": int((sid_fit == 2).sum()),
        "tampered_weight": 5.0,
        "C": args.C,
    }
    if val_path.exists():
        X_val, y_val, _sid_val, _v = load_npz(val_path)
        scores = positive_proba(model, X_val)
        pred = (scores >= 0.5).astype(int)
        metrics.update(
            {
                "n_val": int(len(y_val)),
                "acc": float(accuracy_score(y_val, pred)),
                "auc": float(roc_auc_score(y_val, scores)),
                "fpr95": fpr_at_tpr(y_val, scores),
            }
        )
        print(json.dumps(metrics, indent=2))
    else:
        print("No val features; fitting train only")

    meta = {
        "clip_model_id": cfg["clip_model_id"],
        "use_forensic": cfg["use_forensic"],
        "use_tta": cfg["use_tta"],
        "tta_jpeg_quality": cfg["tta_jpeg_quality"],
        "tta_resize_scale": cfg["tta_resize_scale"],
        "seed": cfg["seed"],
        "metrics": metrics,
        "feature_dim": int(X_train.shape[1]),
        "param_note": "Frozen CLIP ViT-B/32 + logistic regression, well under 2B params",
        "training_data": (
            "SID-Set subset. Binary 0=real vs 1=AIGC-positive (full-synthetic + tampered). "
            "Paired official degradations + consistency means. Tampered rows upweighted. "
            "Features via embed_for_score TTA. WildFake unused."
        ),
        "score_path": "embed_for_score",
        "use_tta_for_fit": True,
        "tampered_weight": 5.0,
    }
    out = model_path(cfg)
    save_bundle(out, model, meta)
    note = cfg["artifacts_dir"] / "WEIGHTS.txt"
    note.write_text(
        (
            "SID-SET SUBSET TRAINED (not smoke)\n\n"
            f"n_train={metrics.get('n_train')}\n"
            f"n_val={metrics.get('n_val')}\n"
            f"acc={metrics.get('acc')}\n"
            f"auc={metrics.get('auc')}\n"
            "Encoder: frozen openai/clip-vit-base-patch32 (ViT-B/32, well under 2B).\n"
            "Labels: SID-Set 0=real, 1=AIGC-positive (includes tampered/label 2). WildFake unused.\n"
            "Paired official JPEG/blur/resize views + consistency means. Tampered weight=5.\n"
            "Fit on embed_for_score TTA (same as infer). GPU CLIP when CUDA is available.\n"
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
