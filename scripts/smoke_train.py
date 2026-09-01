"""Fit a tiny sanity-check classifier on samples/. Replace with SID-Set training."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import random_train_augment
from src.config import load_config, model_path
from src.features import DEFAULT_FEATURE_VARIANT, embed_one, feature_dim_for_variant
from src.io_utils import enable_utf8_stdout, list_images, open_image
from src.model import fit_classifier, save_bundle

# Smoke path skips CV A/B (only 2 images); default to the same pre-projection variant
# the real train.py usually selects so the shipped meta stays self-consistent.
SMOKE_VARIANT = DEFAULT_FEATURE_VARIANT


def main() -> None:
    enable_utf8_stdout()
    cfg = load_config()
    paths = list_images(ROOT / "samples")
    if len(paths) < 2:
        raise SystemExit("Run python scripts/make_samples.py first")
    rng = random.Random(cfg["seed"])
    Xs, ys = [], []
    for path in paths:
        image = open_image(path)
        label = 0 if "real" in path.name else 1
        for k in range(4):
            view = image if k == 0 else random_train_augment(image, rng)
            Xs.append(
                embed_one(
                    view,
                    cfg["clip_model_id"],
                    use_forensic=True,
                    use_tta=False,
                    variant=SMOKE_VARIANT,
                )
            )
            ys.append(label)
    model = fit_classifier(np.stack(Xs), np.array(ys), C=1.0)
    save_bundle(
        model_path(cfg),
        model,
        {
            "clip_model_id": cfg["clip_model_id"],
            "use_forensic": True,
            "use_tta": True,
            "feature_variant": SMOKE_VARIANT,
            "feature_dim": int(feature_dim_for_variant(SMOKE_VARIANT, True)),
            "calibration": "none",
            "note": "SMOKE weights from samples only. Replace after scripts/train.py.",
        },
    )
    print(f"Wrote smoke weights to {model_path(cfg)}")


if __name__ == "__main__":
    main()
