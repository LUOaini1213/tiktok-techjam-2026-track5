from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_classifier(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Pipeline:
    model = build_classifier(C=C)
    model.fit(X, y)
    return model


def save_bundle(path: Path, model: Pipeline, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing weights at {path}. Smoke fixture: run scripts/smoke_train.py. Real model: scripts/train.py after SID-Set subset download. Never train on WildFake demo ids."
        )
    return joblib.load(path)


def predict_proba(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    model: Pipeline = bundle["model"]
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]
