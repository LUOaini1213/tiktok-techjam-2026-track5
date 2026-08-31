"""Single embed/score path shared by extract, train metrics, and infer."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .features import embed_one
from .model import predict_proba


def embed_for_score(
    image: Image.Image,
    model_id: str,
    use_forensic: bool = True,
    use_tta: bool = True,
    tta_jpeg_quality: int = 70,
    tta_resize_scale: float = 0.5,
) -> np.ndarray:
    """Inference-aligned embedding. Default TTA is on (clean + JPEG-70 + 0.5×)."""
    return embed_one(
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
