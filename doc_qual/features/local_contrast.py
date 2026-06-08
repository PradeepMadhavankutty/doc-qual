"""Local text/background contrast detector.

Global brightness (handled by brightness.py) misses cases where the image
is neither too dark nor too light overall, but the *local* contrast between
ink and paper is insufficient — faded ink, heavily recycled paper, or toner
spreading.  This module measures the Michelson contrast in local blocks.
"""

from __future__ import annotations

import numpy as np

from doc_qual.utils import linear_score


def local_contrast_features(
    gray: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure local text-to-background contrast.

    Strategy:
    - Divide the image into non-overlapping 64 × 64 px blocks.
    - For each block compute Michelson contrast:
        C = (I_max − I_min) / (I_max + I_min + ε)
    - Report the mean and 10th-percentile contrast across all blocks.
    - Low 10th-percentile (worst-region contrast) is the most predictive
      signal for OCR failure due to insufficient ink contrast.

    Args:
        gray: Grayscale uint8 image.

    Returns:
        (raw_features, normalized_scores).  ``local_contrast`` score:
        100 = high contrast everywhere; 0 = uniformly washed-out.
    """
    h, w = gray.shape
    block = 64

    contrasts: list[float] = []
    for y in range(0, h - block // 2, block):
        for x in range(0, w - block // 2, block):
            patch = gray[y : y + block, x : x + block]
            if patch.size == 0:
                continue
            i_max = float(patch.max())
            i_min = float(patch.min())
            denom = i_max + i_min + 1e-6
            contrasts.append((i_max - i_min) / denom)

    if not contrasts:
        return {"local_contrast_mean": 0.0, "local_contrast_p10": 0.0}, {
            "local_contrast": 0.0
        }

    arr = np.array(contrasts, dtype=np.float32)

    # Apply CLAHE-inspired local normalization check: how many blocks have
    # meaningful structure (contrast > 0.05)?
    active_ratio = float(np.mean(arr > 0.05))

    mean_c = float(np.mean(arr))
    p10_c = float(np.percentile(arr, 10))

    raw: dict[str, float] = {
        "local_contrast_mean": round(mean_c, 4),
        "local_contrast_p10": round(p10_c, 4),
        "local_contrast_active_ratio": round(active_ratio, 4),
    }

    # Scoring bounds: p10 < 0.05 = near-zero contrast blocks (severe).
    # p10 > 0.35 = good ink-paper separation.
    p10_score = linear_score(p10_c, 0.04, 0.38)
    mean_score = linear_score(mean_c, 0.10, 0.55)
    score = 0.65 * p10_score + 0.35 * mean_score

    normalized: dict[str, float] = {"local_contrast": round(score, 2)}
    return raw, normalized
