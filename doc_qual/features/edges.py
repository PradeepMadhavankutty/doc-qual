"""Edge density metrics for text-like structure.

Edge density is computed per text crop and averaged, which gives a much
more stable estimate than running Canny over the whole page (where large
white margins drag the density toward zero).  The piecewise normalization
curve preserves the historical penalty for documents that are either
nearly empty or unusually busy.  When no text crops are detected the
helper falls back to a full-image edge density.

The optional ``crop_metrics`` keyword lets the composite scorer share a
single crop-detection pass between this feature and
:mod:`doc_qual.features.blurriness`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from doc_qual.features.text_crops import calculate_document_blur_edge_metrics
from doc_qual.utils import clamp


def edge_features(
    gray: np.ndarray,
    *,
    crop_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute averaged crop-level edge density and normalize it for OCR readiness.

    Args:
        gray: Grayscale document image.
        crop_metrics: Optional precomputed result of
            :func:`calculate_document_blur_edge_metrics`.  When supplied
            the cached crop detection is reused; otherwise it is computed.

    Returns:
        ``(raw_features, normalized_scores)``.  ``raw_features`` keeps the
        historical ``"edge_density"`` key (now the crop-averaged density,
        with a full-image fallback) plus diagnostic counts:

        * ``edge_crop_count`` — number of valid crops used
        * ``edge_fallback_used`` — ``1.0`` when the full-image fallback fired
    """
    if crop_metrics is None:
        crop_metrics = calculate_document_blur_edge_metrics(gray)
    density = float(crop_metrics["document_edge_score"])

    if density < 0.01:
        score = density / 0.01 * 45.0
    elif density <= 0.12:
        score = 65.0 + (min(density, 0.06) - 0.01) / 0.05 * 35.0
        if density > 0.06:
            score = 100.0 - (density - 0.06) / 0.06 * 25.0
    else:
        score = max(30.0, 75.0 - (density - 0.12) / 0.12 * 45.0)

    raw_features: dict[str, float] = {
        "edge_density": density,
        "edge_crop_count": float(crop_metrics["valid_crop_count"]),
        "edge_fallback_used": 1.0 if crop_metrics["fallback_used"] else 0.0,
    }
    return raw_features, {"edges": clamp(score)}
