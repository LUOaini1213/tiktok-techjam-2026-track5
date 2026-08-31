from __future__ import annotations

import json
from pathlib import Path

from .augment import jpeg_compress
from .config import load_config, model_path
from .data import records_from_folders
from .features import DEFAULT_FEATURE_VARIANT
from .io_utils import open_image
from .model import load_bundle
from .score import embed_for_score, score_from_embed


def collect_errors(k: int = 8, max_images: int | None = None, use_tta: bool | None = None) -> Path:
    cfg = load_config()
    bundle = load_bundle(model_path(cfg))
    meta = bundle["meta"]
    variant = meta.get("feature_variant", DEFAULT_FEATURE_VARIANT)
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == "val"]
    if max_images:
        from .eval_robustness import balanced_val_slice

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
