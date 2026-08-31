"""Extract CLIP+forensic features. Resume-safe npz per split.

Usage:
  python scripts/extract_features.py --split train
  python scripts/extract_features.py --split val --augment
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import multi_paired_official_views
from src.config import load_config
from src.data import records_from_folders
from src.features import FEATURE_VARIANTS, get_device
from src.io_utils import open_image
from src.score import embed_for_score_dual
from src.train_signal import sid_label_from_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument(
        "--augment",
        action="store_true",
        help="Add official JPEG+blur+resize views per train image",
    )
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def extract_split(cfg, split: str, augment: bool, batch_size: int) -> Path:
    device = get_device()
    print(f"embed_device={device} name={device.type if device.type=='cpu' else __import__('torch').cuda.get_device_name(0)}")
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == split]
    if not rows:
        raise SystemExit(f"No images under {cfg['data_dir']}/{split}/{{real,aigc}}")

    rng = random.Random(cfg["seed"])
    feat_chunks = {v: [] for v in FEATURE_VARIANTS}  # cache BOTH pre + proj variants
    labels = []
    sids = []
    batch_imgs = []
    batch_y = []
    batch_sid = []
    views_per_image = 4 if (augment and split == "train") else 1

    use_tta = bool(cfg.get("use_tta", True))
    print(f"score_path=embed_for_score_dual use_tta={use_tta} views_per_image={views_per_image}")

    def flush():
        if not batch_imgs:
            return
        duals = [
            embed_for_score_dual(
                im,
                model_id=cfg["clip_model_id"],
                use_forensic=cfg["use_forensic"],
                use_tta=use_tta,
                tta_jpeg_quality=cfg["tta_jpeg_quality"],
                tta_resize_scale=cfg["tta_resize_scale"],
            )
            for im in batch_imgs
        ]
        for v in FEATURE_VARIANTS:
            feat_chunks[v].append(np.stack([d[v] for d in duals]))
        labels.extend(batch_y)
        sids.extend(batch_sid)
        batch_imgs.clear()
        batch_y.clear()
        batch_sid.clear()

    for row in tqdm(rows, desc=f"feat-{split}"):
        im = open_image(Path(row["path"]))
        sid = sid_label_from_path(row["path"])
        if augment and split == "train":
            for view, _name in multi_paired_official_views(im, rng):
                batch_imgs.append(view)
                batch_y.append(row["label"])
                batch_sid.append(sid)
        else:
            batch_imgs.append(im)
            batch_y.append(row["label"])
            batch_sid.append(sid)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    feats = {
        v: (np.concatenate(feat_chunks[v], axis=0) if feat_chunks[v] else np.zeros((0, 1)))
        for v in FEATURE_VARIANTS
    }
    y = np.array(labels, dtype=np.int64)
    sid_arr = np.array(sids, dtype=np.int64)
    out = cfg["artifacts_dir"] / f"features_{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # X (== X_proj) kept for backward compatibility with any old reader; X_pre/X_proj new.
    np.savez_compressed(
        out,
        X=feats["proj"],
        X_pre=feats["pre"],
        X_proj=feats["proj"],
        y=y,
        sid=sid_arr,
        views_per_image=np.array([views_per_image]),
    )
    print(
        f"Saved pre={feats['pre'].shape} proj={feats['proj'].shape} "
        f"sid_tampered={(sid_arr == 2).sum()} -> {out}"
    )
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config()
    batch = args.batch_size or cfg["batch_size"]
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        extract_split(cfg, split, augment=args.augment, batch_size=batch)


if __name__ == "__main__":
    main()
