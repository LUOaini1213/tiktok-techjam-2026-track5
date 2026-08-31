"""SID-Set subset download and local folder dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .io_utils import list_images

LABEL_REAL = 0
LABEL_AIGC = 1
LABEL_TAMPERED = 2


def mapped_binary_label(sid_label: int) -> int | None:
    """SID-Set 0 → real (0); 1 full-synthetic and 2 tampered → AIGC-positive (1)."""
    sid_label = int(sid_label)
    if sid_label == LABEL_REAL:
        return LABEL_REAL
    if sid_label in (LABEL_AIGC, LABEL_TAMPERED):
        return LABEL_AIGC
    return None


def folder_for_binary(bin_label: int) -> str:
    return "real" if int(bin_label) == LABEL_REAL else "aigc"


def ingest_sid_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one SID-style dict to a binary training record. Label 2 is kept as AIGC."""
    mapped = mapped_binary_label(record["label"])
    if mapped is None:
        return None
    return {
        "sid_label": int(record["label"]),
        "label": mapped,
        "folder": folder_for_binary(mapped),
        "image": record.get("image"),
    }


def save_split(path: Path, split: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_folders(root: Path) -> list[dict[str, Any]]:
    """Expect data_dir/{train,val}/{real,aigc}/*.jpg"""
    rows = []
    for split in ("train", "val"):
        for name, label in (("real", LABEL_REAL), ("aigc", LABEL_AIGC)):
            folder = root / split / name
            if not folder.exists():
                continue
            for p in list_images(folder):
                rows.append({"path": str(p), "label": label, "split": split})
    return rows


def write_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)


def _sid_quotas(per_binary: int) -> dict[int, int]:
    """Fill binary AIGC from both full-synthetic and tampered."""
    synth = max(1, per_binary // 2)
    tamp = max(1, per_binary - synth)
    return {LABEL_REAL: per_binary, LABEL_AIGC: synth, LABEL_TAMPERED: tamp}


def _existing_sid_counts(out_dir: Path, split_name: str) -> dict[int, int]:
    counts = {LABEL_REAL: 0, LABEL_AIGC: 0, LABEL_TAMPERED: 0}
    for folder, default_sid in (("real", LABEL_REAL), ("aigc", LABEL_AIGC)):
        path = out_dir / split_name / folder
        if not path.exists():
            continue
        for p in list_images(path):
            prefix = p.stem.split("_", 1)[0]
            sid = int(prefix) if prefix.isdigit() else default_sid
            if sid not in counts:
                sid = default_sid
            counts[sid] += 1
    return counts


def subsample_sid_streaming(
    out_dir: Path,
    train_per_class: int,
    val_per_class: int,
    seed: int = 2026,
) -> dict[str, Any]:
    """Stream SID-Set. Label 2 is mapped to AIGC-positive, never dropped."""
    from datasets import load_dataset

    quotas = {
        "train": _sid_quotas(train_per_class),
        "val": _sid_quotas(val_per_class),
    }
    counts = {
        "train": _existing_sid_counts(out_dir, "train"),
        "val": _existing_sid_counts(out_dir, "val"),
    }
    saved = []

    def _full(split_name: str) -> bool:
        return all(counts[split_name][k] >= quotas[split_name][k] for k in quotas[split_name])

    for split_name, hf_split in (("train", "train"), ("val", "validation")):
        if _full(split_name):
            continue
        try:
            ds = load_dataset("saberzl/SID_Set", split=hf_split, streaming=True)
        except Exception:
            ds = load_dataset(
                "saberzl/SID_Set",
                split="val" if split_name == "val" else "train",
                streaming=True,
            )

        for i, row in enumerate(ds):
            ingested = ingest_sid_record(row)
            if ingested is None:
                continue
            sid = ingested["sid_label"]
            if counts[split_name][sid] >= quotas[split_name][sid]:
                if _full(split_name):
                    break
                continue
            img = ingested["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            idx = counts[split_name][sid]
            dest = out_dir / split_name / ingested["folder"] / f"{sid}_{idx:06d}.jpg"
            while dest.exists():
                idx += 1
                dest = out_dir / split_name / ingested["folder"] / f"{sid}_{idx:06d}.jpg"
            write_image(dest, img)
            counts[split_name][sid] = idx + 1
            saved.append(
                {
                    "path": str(dest),
                    "sid_label": sid,
                    "label": ingested["label"],
                    "split": split_name,
                }
            )
            if i % 200 == 0:
                print(
                    f"{split_name} sid0={counts[split_name][0]} "
                    f"sid1={counts[split_name][1]} sid2={counts[split_name][2]} scanned={i}"
                )

    binary = {
        split: {
            "real": counts[split][LABEL_REAL],
            "aigc_positive": counts[split][LABEL_AIGC] + counts[split][LABEL_TAMPERED],
            "tampered": counts[split][LABEL_TAMPERED],
        }
        for split in ("train", "val")
    }
    manifest = {
        "seed": seed,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "counts_by_sid_label": counts,
        "binary_counts": binary,
        "note": (
            "SID-Set binary: 0=real, 1=AIGC-positive (full-synthetic + tampered). "
            "WildFake demo split never used."
        ),
    }
    save_split(out_dir / "split.json", manifest)
    return manifest
