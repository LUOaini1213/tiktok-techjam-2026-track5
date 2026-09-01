"""Cache ONE extra augmentation-family view per training image.

The shipped train cache holds 4 views/image (clean + the three paired official families
jpeg/blur/resize). Gaussian noise is an official Track-5 transform but was never a
training view, and it is by far our weakest robustness row -- this script adds that view
without re-extracting the 4 that already exist.

Row order matches `scripts/extract_features.py` exactly (same `records_from_folders`
order, same split filter), so `scripts/train.py --extra_features` can interleave the
result into each image's view block.

  python scripts/extract_extra_view.py --family noise --shard 0/4 --out artifacts/_extra/p0.npz
  python scripts/extract_extra_view.py --merge artifacts/_extra/p*.npz --out artifacts/features_train_noise.npz
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import (
    BLUR_SIGMAS,
    CENTER_CROP,
    COLOR_JITTER,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    apply_named,
    gaussian_noise,
)
from src.config import load_config
from src.data import records_from_folders
from src.features import FEATURE_VARIANTS
from src.io_utils import enable_utf8_stdout, open_image
from src.score import embed_for_score_dual
from src.train_signal import sid_label_from_path

# Parameter pool per family, drawn per image from a deterministic per-image RNG so the
# classifier sees the whole official severity range rather than one fixed setting.
FAMILY_PARAMS = {
    "noise": NOISE_SIGMAS,
    "jpeg": JPEG_QUALITIES,
    "blur": BLUR_SIGMAS,
    "resize": RESIZE_SCALES,
    "jitter": (COLOR_JITTER,),
    "crop": (CENTER_CROP,),
}


def _stable_seed(*parts) -> int:
    """Deterministic across processes and runs -- unlike hash(), which is salted per process."""
    digest = hashlib.blake2b(":".join(str(x) for x in parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="noise", choices=sorted(FAMILY_PARAMS))
    p.add_argument("--split", default="train")
    p.add_argument("--shard", default="0/1", help="i/N contiguous row shard")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--merge", nargs="+", type=Path, default=None, help="Merge shard npz files (in order) and exit")
    return p.parse_args()


def merge(parts: list[Path], out: Path) -> Path:
    """Concatenate shard npz files in the order given -- shards are contiguous row ranges."""
    loaded = [np.load(p) for p in parts]
    starts = [int(d["shard_start"][0]) for d in loaded]
    if starts != sorted(starts):
        order = np.argsort(starts)
        loaded = [loaded[i] for i in order]
    covered = 0
    for d in loaded:
        if int(d["shard_start"][0]) != covered:
            raise SystemExit(f"Shard gap/overlap: expected start {covered}, got {int(d['shard_start'][0])}")
        covered += len(d["y"])
    total = int(loaded[0]["n_rows_total"][0])
    if covered != total:
        raise SystemExit(f"Shards cover {covered} of {total} images -- refusing to write a short cache")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{f"X_{v}": np.concatenate([d[f"X_{v}"] for d in loaded], axis=0) for v in FEATURE_VARIANTS},
        y=np.concatenate([d["y"] for d in loaded]),
        sid=np.concatenate([d["sid"] for d in loaded]),
        views_per_image=np.array([1]),
    )
    print(f"Wrote {out} ({covered} rows)")
    return out


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    cfg = load_config()
    if args.merge:
        merge(args.merge, args.out)
        return

    i_shard, n_shard = (int(x) for x in args.shard.split("/"))
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"No images for split={args.split} under {cfg['data_dir']}")

    bounds = np.linspace(0, len(rows), n_shard + 1).astype(int)
    lo, hi = int(bounds[i_shard]), int(bounds[i_shard + 1])
    params = FAMILY_PARAMS[args.family]
    feats = {v: [] for v in FEATURE_VARIANTS}
    labels, sids = [], []

    for i in tqdm(range(lo, hi), desc=f"{args.family}[{i_shard}/{n_shard}]"):
        row = rows[i]
        # Independent of the 4 cached views' RNG stream, but still deterministic per image.
        rng = random.Random(f"{cfg['seed']}:{args.split}:extra:{args.family}:{i}")
        param = rng.choice(params)
        img = open_image(Path(row["path"]))
        if args.family == "noise":
            # Seed the noise draw so a re-extraction reproduces this cache byte-for-byte.
            view = gaussian_noise(img, param, np.random.default_rng(_stable_seed(cfg["seed"], args.split, i)))
        else:
            view = apply_named(img, args.family, param)
        dual = embed_for_score_dual(
            view,
            model_id=cfg["clip_model_id"],
            use_forensic=cfg["use_forensic"],
            use_tta=bool(cfg.get("use_tta", True)),
            tta_jpeg_quality=cfg["tta_jpeg_quality"],
            tta_resize_scale=cfg["tta_resize_scale"],
        )
        for v in FEATURE_VARIANTS:
            feats[v].append(dual[v])
        labels.append(int(row["label"]))
        sids.append(sid_label_from_path(row["path"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **{f"X_{v}": np.stack(feats[v]) for v in FEATURE_VARIANTS},
        y=np.array(labels, dtype=np.int64),
        sid=np.array(sids, dtype=np.int64),
        views_per_image=np.array([1]),
        shard_start=np.array([lo]),
        n_rows_total=np.array([len(rows)]),
    )
    print(f"Wrote {args.out} rows={hi - lo} range=[{lo},{hi})")


if __name__ == "__main__":
    main()
