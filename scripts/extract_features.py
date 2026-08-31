"""Extract CLIP+forensic features (both pre/proj variants). Resume-safe npz per split.

The long CPU run is checkpointed: every `--checkpoint_every` source images the
accumulated features are atomically written to features_<split>.partial.npz, and a
fresh run resumes from that partial instead of starting over (a killed 60-minute run
previously lost everything because the npz was only written at the end). Augment views
use a PER-IMAGE seeded RNG so the extracted features are identical no matter where a
resume happened.

Usage:
  python scripts/extract_features.py --split train --augment
  python scripts/extract_features.py --split val
"""

from __future__ import annotations

import argparse
import os
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
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=150,
        help="Write a resumable partial npz every N source images",
    )
    return p.parse_args()


def _atomic_savez(path: Path, **arrays) -> None:
    """Write npz to a temp file then os.replace so a mid-write kill never corrupts it."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_partial(part_path: Path, views_per_image: int, n_rows_total: int):
    """Return (chunks, labels, sids, start_image) from a compatible partial, else fresh."""
    fresh = ({v: [] for v in FEATURE_VARIANTS}, [], [], 0)
    if not part_path.exists():
        return fresh
    try:
        d = np.load(part_path)
        if int(d["views_per_image"][0]) != views_per_image:
            print("partial has different views_per_image; restarting split from scratch")
            return fresh
        n_done = int(d["n_images_done"][0])
        if n_done <= 0 or n_done > n_rows_total:
            return fresh
        chunks = {v: [d[f"X_{v}"]] for v in FEATURE_VARIANTS}
        labels = d["y"].tolist()
        sids = d["sid"].tolist()
        print(f"resuming from partial checkpoint: {n_done} images already embedded")
        return chunks, labels, sids, n_done
    except Exception as exc:
        print(f"could not read partial ({exc!r}); restarting split from scratch")
        return fresh


def extract_split(cfg, split: str, augment: bool, checkpoint_every: int = 150) -> Path:
    device = get_device()
    print(f"embed_device={device}")
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == split]
    if not rows:
        raise SystemExit(f"No images under {cfg['data_dir']}/{split}/{{real,aigc}}")

    views_per_image = 4 if (augment and split == "train") else 1
    use_tta = bool(cfg.get("use_tta", True))
    print(f"score_path=embed_for_score_dual use_tta={use_tta} views_per_image={views_per_image}")

    out = cfg["artifacts_dir"] / f"features_{split}.npz"
    part_path = cfg["artifacts_dir"] / f"features_{split}.partial.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    feat_chunks, labels, sids, start = _load_partial(part_path, views_per_image, len(rows))

    def checkpoint(n_images_done: int) -> None:
        _atomic_savez(
            part_path,
            **{f"X_{v}": np.concatenate(feat_chunks[v], axis=0) for v in FEATURE_VARIANTS},
            y=np.array(labels, dtype=np.int64),
            sid=np.array(sids, dtype=np.int64),
            views_per_image=np.array([views_per_image]),
            n_images_done=np.array([n_images_done]),
        )

    for i in tqdm(range(start, len(rows)), initial=start, total=len(rows), desc=f"feat-{split}"):
        row = rows[i]
        im = open_image(Path(row["path"]))
        sid = sid_label_from_path(row["path"])
        # Per-image RNG: augment views for image i are the same whether or not a resume
        # happened before it (a shared sequential RNG would desync on resume).
        rng = random.Random(f"{cfg['seed']}:{split}:{i}")
        if augment and split == "train":
            views = [v for v, _name in multi_paired_official_views(im, rng)]
        else:
            views = [im]
        duals = [
            embed_for_score_dual(
                v,
                model_id=cfg["clip_model_id"],
                use_forensic=cfg["use_forensic"],
                use_tta=use_tta,
                tta_jpeg_quality=cfg["tta_jpeg_quality"],
                tta_resize_scale=cfg["tta_resize_scale"],
            )
            for v in views
        ]
        for v in FEATURE_VARIANTS:
            feat_chunks[v].append(np.stack([d[v] for d in duals]))
        labels.extend([row["label"]] * len(views))
        sids.extend([sid] * len(views))

        done = i + 1
        if done < len(rows) and (done - start) % checkpoint_every == 0:
            checkpoint(done)

    feats = {
        v: (np.concatenate(feat_chunks[v], axis=0) if feat_chunks[v] else np.zeros((0, 1)))
        for v in FEATURE_VARIANTS
    }
    y = np.array(labels, dtype=np.int64)
    sid_arr = np.array(sids, dtype=np.int64)
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
    if part_path.exists():
        part_path.unlink()
    print(
        f"Saved pre={feats['pre'].shape} proj={feats['proj'].shape} "
        f"sid_tampered={(sid_arr == 2).sum()} -> {out}"
    )
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config()
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        extract_split(cfg, split, augment=args.augment, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
