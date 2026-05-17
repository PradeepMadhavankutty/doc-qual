"""Edge density metrics for text-like structure."""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import clamp


def edge_features(gray: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Compute Canny edge density and normalize it for OCR readiness."""

    edges = cv2.Canny(gray, 80, 180)
    density = float(np.count_nonzero(edges) / edges.size)

    if density < 0.01:
        score = density / 0.01 * 45.0
    elif density <= 0.12:
        score = 65.0 + (min(density, 0.06) - 0.01) / 0.05 * 35.0
        if density > 0.06:
            score = 100.0 - (density - 0.06) / 0.06 * 25.0
    else:
        score = max(30.0, 75.0 - (density - 0.12) / 0.12 * 45.0)

    return {"edge_density": density}, {"edges": clamp(score)}
