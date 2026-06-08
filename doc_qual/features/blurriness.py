"""Blur estimation using Laplacian variance.

Blur is estimated by detecting text-like crops and averaging the
Laplacian variance of each valid crop.  This focuses the metric on the
regions OCR actually needs to read instead of being diluted by large
white margins.  When no text crops are detected the helper falls back to
the full-image Laplacian variance so the API never returns a zero score
for legitimate but hard-to-segment images.

The optional ``crop_metrics`` keyword lets the composite scorer share a
single crop-detection pass between this feature and
:mod:`doc_qual.features.edges`, avoiding duplicate work on the same
image.  When called standalone the function recomputes the metrics
itself, preserving the original one-argument call signature.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from doc_qual.features.text_crops import calculate_document_blur_edge_metrics
from doc_qual.utils import linear_score


def blurriness_features(
    gray: np.ndarray,
    *,
    crop_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute blur-related raw values and normalized scores.

    Args:
        gray: Grayscale document image.
        crop_metrics: Optional precomputed result of
            :func:`calculate_document_blur_edge_metrics`.  Pass this when
            the caller already ran crop detection for another feature to
            avoid recomputing it.

    Returns:
        ``(raw_features, normalized_scores)`` where ``raw_features`` keeps
        the historical ``"laplacian_variance"`` key (now the crop-average
        Laplacian variance, falling back to the full-image value when no
        text crops are found) and ``normalized_scores`` provides a single
        ``"sharpness"`` score on the existing 0-100 scale.

        Additional informational keys in ``raw_features``:

        * ``blur_crop_count`` — number of valid crops used
        * ``blur_fallback_used`` — ``1.0`` when the full-image fallback fired
    """
    if crop_metrics is None:
        crop_metrics = calculate_document_blur_edge_metrics(gray)
    variance = float(crop_metrics["document_blur_score"])
    score = linear_score(variance, 35.0, 650.0)
    raw_features: dict[str, float] = {
        "laplacian_variance": variance,
        "blur_crop_count": float(crop_metrics["valid_crop_count"]),
        "blur_fallback_used": 1.0 if crop_metrics["fallback_used"] else 0.0,
    }
    return raw_features, {"sharpness": score}
