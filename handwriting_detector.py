"""
ScanMind handwriting detector.

It supports two modes:
1) Trained MobileNetV3 classifier, when HANDWRITING_MODEL_PATH points to a
   trained .pt checkpoint.
2) OpenCV baseline fallback, so the API can be tested before the classifier
   is trained.

The classifier predicts:
    0 = handwritten
    1 = printed

The training script creates the expected checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = os.getenv("HANDWRITING_MODEL_PATH", "").strip()

_model = None
_device = None
_model_error = None


def _load_trained_model():
    global _model, _device, _model_error

    if _model is not None or _model_error is not None:
        return _model

    if not MODEL_PATH:
        _model_error = "HANDWRITING_MODEL_PATH is not set"
        return None

    path = Path(MODEL_PATH)
    if not path.exists():
        _model_error = f"Model not found: {path}"
        return None

    try:
        import torch
        from torchvision import models

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        import torch.nn as nn
        model.classifier[-1] = nn.Linear(in_features, 2)

        checkpoint = torch.load(path, map_location=_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(_device)

        _model = model
        return _model
    except Exception as exc:
        _model_error = str(exc)
        return None


def _baseline_features(image: np.ndarray) -> dict:
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    h, w = gray.shape[:2]
    scale = min(1200 / max(h, 1), 1200 / max(w, 1), 1.0)
    if scale != 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
        )

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    components = stats[1:, cv2.CC_STAT_AREA]
    components = components[components >= 8]
