"""Training-signal helpers: tampered reweight + clean/degraded consistency rows.

No CLIP. Operates on numpy features so tests stay fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TAMPERED_SID = 2
DEFAULT_TAMPERED_WEIGHT = 5.0


def sid_label_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    prefix = stem.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return 1 if Path(path).parent.name == "aigc" else 0


def tampered_sample_weights(
    sid_labels: np.ndarray,
    tampered_weight: float = DEFAULT_TAMPERED_WEIGHT,
) -> np.ndarray:
    """SID label 2 (tampered) is upweighted vs other rows."""
    sid = np.asarray(sid_labels, dtype=np.int64)
    weights = np.ones(sid.shape[0], dtype=np.float64)
    weights[sid == TAMPERED_SID] = float(tampered_weight)
    return weights


def expand_with_consistency(
    X: np.ndarray,
    y: np.ndarray,
    sid_labels: np.ndarray,
    views_per_image: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append mean(clean, degraded) rows for each image's view block."""
    if views_per_image < 2:
        return X, y, sid_labels
    n = X.shape[0]
    if n % views_per_image != 0:
        raise ValueError("feature rows must be a multiple of views_per_image")
    n_img = n // views_per_image
    extras_X = []
    extras_y = []
    extras_sid = []
    for i in range(n_img):
        sl = slice(i * views_per_image, (i + 1) * views_per_image)
        block = X[sl]
        clean = block[0]
        for deg in block[1:]:
            extras_X.append(0.5 * (clean + deg))
            extras_y.append(y[sl][0])
            extras_sid.append(sid_labels[sl][0])
    X2 = np.concatenate([X, np.stack(extras_X)], axis=0)
    y2 = np.concatenate([y, np.asarray(extras_y, dtype=y.dtype)])
    s2 = np.concatenate([sid_labels, np.asarray(extras_sid, dtype=sid_labels.dtype)])
    return X2, y2, s2


def fit_with_training_signal(
    X: np.ndarray,
    y: np.ndarray,
    sid_labels: np.ndarray,
    *,
    views_per_image: int = 2,
    tampered_weight: float = DEFAULT_TAMPERED_WEIGHT,
    C: float = 1.0,
) -> Pipeline:
    X2, y2, s2 = expand_with_consistency(X, y, sid_labels, views_per_image)
    weights = tampered_sample_weights(s2, tampered_weight=tampered_weight)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X2, y2, clf__sample_weight=weights)
    return pipe


def positive_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]
