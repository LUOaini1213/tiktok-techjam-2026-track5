"""Single embed/score path shared by extract, train metrics, and infer."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .features import DEFAULT_FEATURE_VARIANT, embed_one, embed_one_dual
from .model import predict_proba


def embed_for_score(
    image: Image.Image,
    model_id: str,
    use_forensic: bool = True,
    use_tta: bool = True,
    tta_jpeg_quality: int = 70,
    tta_resize_scale: float = 0.5,
    variant: str = DEFAULT_FEATURE_VARIANT,
    crops: int = 0,
) -> np.ndarray:
    """Inference-aligned single-variant embedding. Default TTA on (clean + JPEG-70 + 0.5x).

    `crops=N` adds N corner/centre crops to the averaged view set (eval-time multi-crop).
    """
    return embed_one(
        image,
        model_id=model_id,
        use_forensic=use_forensic,
        use_tta=use_tta,
        tta_jpeg_quality=tta_jpeg_quality,
        tta_resize_scale=tta_resize_scale,
        variant=variant,
        crops=crops,
    )


def embed_for_score_dual(
    image: Image.Image,
    model_id: str,
    use_forensic: bool = True,
    use_tta: bool = True,
    tta_jpeg_quality: int = 70,
    tta_resize_scale: float = 0.5,
) -> dict[str, np.ndarray]:
    """Both variants ({"pre","proj"}) from one shared embed pass -- used by feature cache."""
    return embed_one_dual(
        image,
        model_id=model_id,
        use_forensic=use_forensic,
        use_tta=use_tta,
        tta_jpeg_quality=tta_jpeg_quality,
        tta_resize_scale=tta_resize_scale,
    )


def score_from_embed(bundle: dict[str, Any], feat: np.ndarray) -> float:
    pred = float(predict_proba(bundle, feat.reshape(1, -1))[0])
    return min(1.0, max(0.0, pred))
