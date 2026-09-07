"""Extract fused features for every variant (pre / proj / dino / fuse). Resume-safe, shardable.

Two protections for a long CPU run:

* Checkpointing -- every `--checkpoint_every` source images the accumulated features are
  atomically written to a .partial.npz and a fresh run resumes from it. Augment views use a
  PER-IMAGE seeded RNG, so the extracted features are identical no matter where a resume
  happened.
* Sharding -- `--shard i/N` embeds a contiguous row range into its own npz; `--merge` joins
  the shards in order and refuses to write a short cache. One process per core-group is
  the only way a 16k-image, 6-view extraction fits in an afternoon on a CPU box.

Usage:
  python scripts/extract_features.py --split train --augment --shard 0/5 --out artifacts/_feat/train_0.npz
  python scripts/extract_features.py --merge artifacts/_feat/train_*.npz --out artifacts/features_train.npz
  python scripts/extract_features.py --split val                       # single process, no views
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

from src.augment import PAIR_FAMILIES, multi_paired_official_views
from src.config import load_config
from src.data import records_from_folders
from src.features import FEATURE_VARIANTS, get_device
from src.io_utils import enable_utf8_stdout, open_image
from src.score import embed_for_score_dual
from src.train_signal import sid_label_from_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--augment", action="store_true", help="One degraded view per official family for train")
    p.add_argument("--checkpoint_every", type=int, default=150, help="Resumable partial npz every N images")
    p.add_argument("--shard", default=None, help="i/N contiguous row shard (requires --out)")
    p.add_argument("--out", type=Path, default=None, help="Output npz (default artifacts/features_<split>.npz)")
    p.add_argument("--merge", nargs="+", type=Path, default=None, help="Merge shard npz files into --out")
    return p.parse_args()


def _atomic_savez(path: Path, **arrays) -> None:
    """Write npz to a temp file then os.replace so a mid-write kill never corrupts it."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_partial(part_path: Path, views_per_image: int, n_rows_total: int, lo: int):
    """Return (chunks, labels, sids, next_image) from a compatible partial, else fresh."""
    fresh = ({v: [] for v in FEATURE_VARIANTS}, [], [], lo)
    if not part_path.exists():
        return fresh
    try:
        d = np.load(part_path)
        if int(d["views_per_image"][0]) != views_per_image or int(d["n_rows_total"][0]) != n_rows_total:
            print("partial has a different layout; restarting this shard from scratch")
            return fresh
        if any(f"X_{v}" not in d.files for v in FEATURE_VARIANTS):
            print("partial lacks a feature variant; restarting this shard from scratch")
            return fresh
        n_done = int(d["n_images_done"][0])
        chunks = {v: [d[f"X_{v}"]] for v in FEATURE_VARIANTS}
        print(f"resuming from partial checkpoint: {n_done - lo} images already embedded")
        return chunks, list(d["y"]), list(d["sid"]), n_done
    except Exception as exc:  # noqa: BLE001
        print(f"could not read partial ({exc!r}); restarting this shard from scratch")
        return fresh


def extract_split(cfg, split: str, augment: bool, checkpoint_every: int, shard: str | None, out: Path | None) -> Path:
    device = get_device()
    rows = [r for r in records_from_folders(cfg["data_dir"]) if r["split"] == split]
    if not rows:
        raise SystemExit(f"No images under {cfg['data_dir']}/{split}/{{real,aigc}}")

    i_shard, n_shard = (0, 1)
    if shard:
        i_shard, n_shard = (int(x) for x in shard.split("/"))
        if out is None:
            raise SystemExit("--shard requires --out")
    bounds = np.linspace(0, len(rows), n_shard + 1).astype(int)
    lo, hi = int(bounds[i_shard]), int(bounds[i_shard + 1])

    views_per_image = (1 + len(PAIR_FAMILIES)) if (augment and split == "train") else 1
    use_tta = bool(cfg.get("use_tta", True))
    print(f"embed_device={device} split={split} rows=[{lo},{hi}) of {len(rows)} views_per_image={views_per_image} "
          f"variants={FEATURE_VARIANTS} use_tta={use_tta}")

    out = out or (cfg["artifacts_dir"] / f"features_{split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    part_path = out.with_suffix(".partial.npz")

    feat_chunks, labels, sids, start = _load_partial(part_path, views_per_image, len(rows), lo)

    def checkpoint(n_images_done: int) -> None:
        _atomic_savez(
            part_path,
            **{f"X_{v}": np.concatenate(feat_chunks[v], axis=0) for v in FEATURE_VARIANTS},
            y=np.array(labels, dtype=np.int64),
            sid=np.array(sids, dtype=np.int64),
            views_per_image=np.array([views_per_image]),
            n_rows_total=np.array([len(rows)]),
            n_images_done=np.array([n_images_done]),
        )

    for i in tqdm(range(start, hi), initial=start - lo, total=hi - lo, desc=f"feat-{split}[{i_shard}/{n_shard}]"):
        row = rows[i]
        im = open_image(Path(row["path"]))
        sid = sid_label_from_path(row["path"])
        # Per-image RNG: the views for image i are the same whether or not a resume happened.
        rng = random.Random(f"{cfg['seed']}:{split}:{i}")
        views = [v for v, _name in multi_paired_official_views(im, rng)] if views_per_image > 1 else [im]
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
        if done < hi and (done - start) % checkpoint_every == 0:
            checkpoint(done)

    feats = {v: np.concatenate(feat_chunks[v], axis=0) for v in FEATURE_VARIANTS}
    np.savez_compressed(
        out,
        X=feats["proj"],  # legacy single-variant readers
        **{f"X_{v}": feats[v] for v in FEATURE_VARIANTS},
        y=np.array(labels, dtype=np.int64),
        sid=np.array(sids, dtype=np.int64),
        views_per_image=np.array([views_per_image]),
        shard_start=np.array([lo]),
        n_rows_total=np.array([len(rows)]),
    )
    if part_path.exists():
        part_path.unlink()
    print("Saved " + " ".join(f"{v}={feats[v].shape}" for v in FEATURE_VARIANTS) + f" -> {out}")
    return out


def merge(parts: list[Path], out: Path) -> Path:
    """Concatenate shard npz files in row order; refuse to write a short cache."""
    loaded = sorted((np.load(p) for p in parts), key=lambda d: int(d["shard_start"][0]))
    total = int(loaded[0]["n_rows_total"][0])
    vpi = int(loaded[0]["views_per_image"][0])
    covered = 0
    for d in loaded:
        if int(d["shard_start"][0]) != covered:
            raise SystemExit(f"shard gap/overlap: expected start {covered}, got {int(d['shard_start'][0])}")
        if int(d["views_per_image"][0]) != vpi:
            raise SystemExit("shards disagree on views_per_image")
        covered += len(d["y"]) // vpi
    if covered != total:
        raise SystemExit(f"shards cover {covered} of {total} images -- refusing to write a short cache")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats = {v: np.concatenate([d[f"X_{v}"] for d in loaded], axis=0) for v in FEATURE_VARIANTS}
    np.savez_compressed(
        out,
        X=feats["proj"],
        **{f"X_{v}": feats[v] for v in FEATURE_VARIANTS},
        y=np.concatenate([d["y"] for d in loaded]),
        sid=np.concatenate([d["sid"] for d in loaded]),
        views_per_image=np.array([vpi]),
    )
    print(f"Wrote {out}: {covered} images x {vpi} views, " + " ".join(f"{v}={feats[v].shape[1]}-D" for v in FEATURE_VARIANTS))
    return out


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    cfg = load_config()
    if args.merge:
        if args.out is None:
            raise SystemExit("--merge requires --out")
        merge(args.merge, args.out)
        return
    splits = ["train", "val"] if args.split == "both" else [args.split]
    if args.shard and len(splits) != 1:
        raise SystemExit("--shard needs a single --split")
    for split in splits:
        extract_split(cfg, split, augment=args.augment, checkpoint_every=args.checkpoint_every,
                      shard=args.shard, out=args.out)


if __name__ == "__main__":
    main()
