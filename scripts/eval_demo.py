"""WildFake demonstration benchmark (NEVER used for training).

Streams the official demonstration subset (techjam-aigc/wildfake-eval-subset,
`default` config = COCO val2017 reals + DALL-E-3 Advanced fakes), takes a BALANCED
subsample, scores it through the SHIPPED inference path (embed_for_score + the trained
bundle, same variant/TTA as infer.py), and reports acc/AUC on clean plus a few
representative Track-5 transforms. Writes results/demo_benchmark.csv.

This data is a reference benchmark only; it must not touch training.

Usage:
  python scripts/eval_demo.py                 # 1000 real + 1000 fake
  python scripts/eval_demo.py --per_class 500
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import apply_named
from src.config import load_config, model_path
from src.eval_robustness import stratified_bootstrap_auc_ci
from src.features import DEFAULT_FEATURE_VARIANT
from src.io_utils import enable_utf8_stdout
from src.io_utils import open_image  # noqa: F401  (kept for parity / future file inputs)
from src.model import load_bundle
from src.score import embed_for_score, score_from_embed

DATASET = "techjam-aigc/wildfake-eval-subset"
CONFIG = "default"  # COCO val2017 reals (label 0) + dalle3_advanced fakes (label 1)
# (name, op, param) matching src.augment EVAL_PRESETS. None op == clean.
DEMO_TRANSFORMS = [
    ("clean", None, None),
    ("jpeg_30", "jpeg", 30),
    ("resize_0.25", "resize", 0.25),
    ("noise_0.05", "noise", 0.05),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per_class", type=int, default=1000)
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--bootstrap_reps", type=int, default=2000)
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Local cache for the demo subsample (default data/demo_subset; gitignored)",
    )
    p.add_argument(
        "--fetch_only",
        action="store_true",
        help="Only fill the local cache (network phase); skip scoring",
    )
    p.add_argument(
        "--transforms",
        type=str,
        default=None,
        help="Comma-separated DEMO_TRANSFORMS names to score (default: all)",
    )
    p.add_argument("--out", type=Path, default=None, help="Write this CSV instead of results/demo_benchmark.csv")
    p.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        help="Merge these part CSVs into results/demo_benchmark.csv and exit",
    )
    return p.parse_args()


FIELDS = ["transform", "n", "acc", "auc", "auc_lo", "auc_hi"]


def merge_parts(parts, dest: Path) -> Path:
    """Merge sharded demo CSVs in DEMO_TRANSFORMS order (sharding is a CPU trick only)."""
    order = [n for n, _, _ in DEMO_TRANSFORMS]
    seen = {}
    for part in parts:
        if not Path(part).exists():
            continue
        with Path(part).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen[row["transform"]] = row
    missing = [n for n in order if n not in seen]
    if missing:
        raise SystemExit(f"Merged demo table incomplete, missing: {missing}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name in order:
            w.writerow(seen[name])
    return dest


def _cache_paths(cache_dir: Path, per_class: int) -> dict[int, list[Path]]:
    out = {}
    for label in (0, 1):
        out[label] = sorted(cache_dir.glob(f"{label}_*.png"))[:per_class]
    return out


def fill_cache(cache_dir: Path, per_class: int) -> None:
    """Stream the demo subset into a lossless local PNG cache (resume-safe by count).

    PNG preserves the decoded pixels exactly -- re-encoding to JPEG would stack a second
    compression generation on top of the dataset's own and shift the forensic stats.
    Never written anywhere near the training dirs.
    """
    from datasets import load_dataset

    have = {k: len(v) for k, v in _cache_paths(cache_dir, per_class).items()} if cache_dir.exists() else {0: 0, 1: 0}
    if all(have[k] >= per_class for k in have):
        print(f"cache already full: real={have[0]} fake={have[1]}")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(DATASET, CONFIG, split="validation", streaming=True)
    except Exception as exc:
        raise SystemExit(f"Could not stream {DATASET}:{CONFIG} -- skip demo eval. ({exc!r})")

    scanned = 0
    for row in ds:
        scanned += 1
        label = int(row.get("label", -1))
        if label not in have or have[label] >= per_class:
            if all(have[k] >= per_class for k in have):
                break
            continue
        img = row.get("image")
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(np.asarray(img))
            except Exception:
                continue
        img.convert("RGB").save(cache_dir / f"{label}_{have[label]:05d}.png")
        have[label] += 1
        if scanned % 500 == 0:
            print(f"  scanned={scanned} real={have[0]}/{per_class} fake={have[1]}/{per_class}")
    print(f"cache filled: real={have[0]} fake={have[1]} (scanned {scanned})")


def load_balanced(per_class: int, seed: int, cache_dir: Path) -> list[tuple[Image.Image, int]]:
    """Balanced demo sample from the local PNG cache (fills it first if needed)."""
    fill_cache(cache_dir, per_class)
    out: list[tuple[Image.Image, int]] = []
    paths = _cache_paths(cache_dir, per_class)
    for label in (0, 1):
        for p in paths[label]:
            out.append((Image.open(p).convert("RGB"), label))
    print(f"Loaded real={len(paths[0])} fake={len(paths[1])} from {cache_dir}")
    return out


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    cfg = load_config()
    cache_dir = args.cache_dir or (cfg["root"] / "data" / "demo_subset")

    if args.merge:
        dest = args.out or (cfg["results_dir"] / "demo_benchmark.csv")
        print(f"Wrote {merge_parts(args.merge, dest)}")
        return

    if args.fetch_only:
        fill_cache(cache_dir, args.per_class)
        return

    bundle = load_bundle(model_path(cfg))
    meta = bundle["meta"]
    variant = meta.get("feature_variant", DEFAULT_FEATURE_VARIANT)
    use_tta = False if args.no_tta else bool(meta.get("use_tta", cfg["use_tta"]))

    samples = load_balanced(args.per_class, cfg["seed"], cache_dir)
    if not samples:
        raise SystemExit("No demo samples loaded -- skip demo eval.")

    dest = args.out or (cfg["results_dir"] / "demo_benchmark.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    todo = DEMO_TRANSFORMS
    if args.transforms:
        wanted = {t.strip() for t in args.transforms.split(",")}
        unknown = wanted - {n for n, _, _ in DEMO_TRANSFORMS}
        if unknown:
            raise SystemExit(f"Unknown demo transform(s): {sorted(unknown)}")
        todo = [t for t in DEMO_TRANSFORMS if t[0] in wanted]
    rows_out = []
    for name, op, param in todo:
        y_true, scores = [], []
        for img, label in samples:
            view = apply_named(img, op, param)
            feat = embed_for_score(
                view,
                model_id=meta.get("clip_model_id", cfg["clip_model_id"]),
                use_forensic=meta.get("use_forensic", True),
                use_tta=use_tta,
                tta_jpeg_quality=cfg["tta_jpeg_quality"],
                tta_resize_scale=cfg["tta_resize_scale"],
                variant=variant,
            )
            scores.append(score_from_embed(bundle, feat))
            y_true.append(label)
        y = np.array(y_true)
        s = np.array(scores)
        acc = float(accuracy_score(y, (s >= 0.5).astype(int)))
        try:
            auc = float(roc_auc_score(y, s))
            lo, hi = stratified_bootstrap_auc_ci(y, s, reps=args.bootstrap_reps, seed=cfg["seed"])
        except ValueError:
            auc = lo = hi = float("nan")
        print(f"{name:14s} n={len(y)} acc={acc:.4f} auc={auc:.4f} [{lo:.4f},{hi:.4f}]")
        rows_out.append(
            {
                "transform": name,
                "n": len(y),
                "acc": f"{acc:.4f}",
                "auc": f"{auc:.4f}",
                "auc_lo": f"{lo:.4f}",
                "auc_hi": f"{hi:.4f}",
            }
        )

    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
