from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from .augment import EVAL_PRESETS, apply_named
from .config import load_config, model_path
from .data import records_from_folders
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


def evaluate_robustness(
    max_images: int | None = None,
    use_tta: bool | None = None,
    table_path: Path | None = None,
) -> Path:
    cfg = load_config()
    bundle = load_bundle(model_path(cfg))
    meta = bundle["meta"]
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]
    rows = balanced_val_slice(rows, max_images)
    if not rows:
        raise RuntimeError("No val images. Run scripts/download_data.py first.")

    if use_tta is None:
        use_tta = bool(meta.get("use_tta", cfg["use_tta"]))

    dest = Path(table_path) if table_path else cfg["results_dir"] / "robustness_table.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["transform", "n", "acc", "auc"]
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, op, param in EVAL_PRESETS:
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
                )
                scores.append(score_from_embed(bundle, feat))
                y_true.append(int(row["label"]))
            y = np.array(y_true)
            s = np.array(scores)
            hard = (s >= 0.5).astype(int)
            acc = float(accuracy_score(y, hard))
            try:
                auc = float(roc_auc_score(y, s))
            except ValueError:
                auc = float("nan")
            writer.writerow(
                {"transform": name, "n": len(y), "acc": f"{acc:.4f}", "auc": f"{auc:.4f}"}
            )
            f.flush()
    return dest
