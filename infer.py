"""Track 5 submission entry: directory in, JSON out.

Usage:
  python infer.py --input_dir path/to/images --output preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A safe P(AIGC) to emit when an image cannot be read/scored, so a single bad file
# never crashes the run and drops the whole preds.json (== zero score).
FALLBACK_PRED = 0.5

from src.config import load_config, model_path
from src.features import DEFAULT_FEATURE_VARIANT, feature_dim_for_variant
from src.io_utils import enable_utf8_stdout, list_images, open_image
from src.model import load_bundle
from src.score import embed_for_score, score_from_embed


def clamp_pred(pred: float) -> float:
    return min(1.0, max(0.0, float(pred)))


def prediction_row(image_path: str | Path, pred: float) -> dict:
    return {"image_path": str(image_path), "pred": clamp_pred(pred)}


def write_predictions(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def score_directory(
    input_dir: Path,
    output: Path,
    *,
    config: Path | None = None,
    weights: Path | None = None,
    use_tta: bool | None = None,
) -> list[dict]:
    cfg = load_config(config)
    bundle = load_bundle(weights or model_path(cfg))
    meta = bundle.get("meta", {})
    model_id = meta.get("clip_model_id", cfg["clip_model_id"])
    use_forensic = meta.get("use_forensic", cfg["use_forensic"])
    variant = meta.get("feature_variant", DEFAULT_FEATURE_VARIANT)
    if use_tta is None:
        use_tta = bool(meta.get("use_tta", cfg["use_tta"]))

    # Fail fast (before the per-image loop, so it is NOT swallowed by the fallback
    # handler) if the weights were trained on a different feature width than the
    # current pipeline produces for this variant -- otherwise every image would
    # silently score the fallback 0.5.
    expected_dim = feature_dim_for_variant(variant, use_forensic)
    trained_dim = meta.get("feature_dim")
    if trained_dim is not None and int(trained_dim) != expected_dim:
        raise SystemExit(
            f"Weights expect {trained_dim}-D features but the current pipeline produces "
            f"{expected_dim}-D for variant '{variant}' (use_forensic={use_forensic}). "
            "The feature extractor changed -- retrain the head:\n"
            "  python scripts/extract_features.py --split train --augment\n"
            "  python scripts/extract_features.py --split val\n"
            "  python scripts/train.py"
        )

    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        # Never crash on a missing/empty dir: a valid (empty) preds.json beats a zero.
        print(f"WARNING: input_dir is not a folder: {input_dir} -- writing empty preds.", file=sys.stderr)
        write_predictions([], output)
        return []

    rows = []
    for path in list_images(input_dir):
        try:
            feat = embed_for_score(
                open_image(path),
                model_id=model_id,
                use_forensic=use_forensic,
                use_tta=use_tta,
                tta_jpeg_quality=cfg["tta_jpeg_quality"],
                tta_resize_scale=cfg["tta_resize_scale"],
                variant=variant,
            )
            pred = score_from_embed(bundle, feat)
        except Exception:  # one unreadable image must not abort the whole run
            print(f"WARNING: failed to score {path}; emitting fallback pred.", file=sys.stderr)
            traceback.print_exc()
            pred = FALLBACK_PRED
        rows.append(prediction_row(path, pred))  # exactly one row per input file

    write_predictions(rows, output)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIGC confidence scores for every image in a folder.")
    p.add_argument("--input_dir", required=True, type=Path, help="Folder of jpg/png/webp images")
    p.add_argument("--output", required=True, type=Path, help="Output JSON path")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None, help="Override artifacts/repostguard.joblib")
    p.add_argument("--no_tta", action="store_true")
    return p.parse_args()


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    try:
        rows = score_directory(
            args.input_dir,
            args.output,
            config=args.config,
            weights=args.weights,
            use_tta=False if args.no_tta else None,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
