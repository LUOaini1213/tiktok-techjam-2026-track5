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
    if variant in ('v4_local','v4_global'):
        if not use_forensic or crops:
            raise ValueError('V4 local descriptor requires forensic features and crops=0')
        from .features_v4 import embed_v4, center_forensic
        if variant == 'v4_local':
            return embed_v4(image, model_id=model_id, use_tta=use_tta,
                            tta_jpeg_quality=tta_jpeg_quality, tta_resize_scale=tta_resize_scale)
        from .features import _tta_clip_means, dino_embed_batch
        clip=_tta_clip_means(image,model_id,use_tta,tta_jpeg_quality,tta_resize_scale)['proj']
        dino=dino_embed_batch([image])[0]
        dino=dino/(np.linalg.norm(dino)+1e-8)
        return np.concatenate([clip,dino,center_forensic(image)]).astype(np.float32)
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
    if not np.isfinite(feat).all():
        raise ValueError("Non-finite image features")
    pred = float(predict_proba(bundle, feat.reshape(1, -1))[0])
    if not np.isfinite(pred):
        raise ValueError("Non-finite model score")
    return min(1.0, max(0.0, pred))
