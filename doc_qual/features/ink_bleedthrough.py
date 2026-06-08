"""Ink bleed-through / show-through detector.

Bleed-through occurs when ink from the reverse side of a page is visible,
appearing as ghost text or smudging in page margins.  We detect it by
looking for unexpected dark pixel clusters in margin regions where real
text is unlikely to be.
"""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def ink_bleedthrough_features(
    gray: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate ink bleed-through / show-through from the reverse side.

    Strategy:
    - Threshold the image to isolate dark pixels (potential ink).
    - Define margin bands (top/bottom 12 %, left/right 8 % of page).
    - Compute the ratio of dark pixels inside the margin bands.
    - Real text is concentrated in the central content area; bleed-through
      appears as diffuse dark pixels spread across margins.
    - Also measure the spatial variance of dark pixels: bleed-through tends
      to be more uniformly distributed than real text.

    Args:
        gray: Grayscale image array, dtype uint8.

    Returns:
        (raw_features, normalized_scores) where the normalized score
        ``ink_bleedthrough`` is 0 (severe bleed) → 100 (clean).
    """
    h, w = gray.shape

    # Isolate dark ink pixels: pixels more than 40 grey levels below the page
    # white-point (90th-percentile brightness) are considered ink.
    white_point = float(np.percentile(gray, 90))
    ink_threshold = max(20, int(white_point - 40))
    _, binary = cv2.threshold(gray, ink_threshold, 255, cv2.THRESH_BINARY_INV)

    # --- Margin masks ---------------------------------------------------------
    top_h = max(1, int(h * 0.12))
    bot_h = max(1, int(h * 0.12))
    lr_w = max(1, int(w * 0.08))

    margin_mask = np.zeros_like(binary)
    margin_mask[:top_h, :] = 255  # top band
    margin_mask[h - bot_h :, :] = 255  # bottom band
    margin_mask[:, :lr_w] = 255  # left band
    margin_mask[:, w - lr_w :] = 255  # right band

    total_margin_px = int(np.sum(margin_mask > 0))
    dark_in_margin = int(np.sum((binary > 0) & (margin_mask > 0)))

    bleedthrough_ratio = dark_in_margin / max(total_margin_px, 1)

    # Spatial uniformity of all dark pixels (low variance = diffuse = bleed)
    ys, xs = np.where(binary > 0)
    if len(xs) > 10:
        norm_xs = xs / max(w, 1)
        norm_ys = ys / max(h, 1)
        spatial_std = float(np.std(norm_xs) + np.std(norm_ys))
    else:
        spatial_std = 1.0  # no dark pixels → clean

    raw: dict[str, float] = {
        "bleedthrough_ratio": round(bleedthrough_ratio, 6),
        "bleedthrough_spatial_std": round(spatial_std, 4),
    }

    # Score: high ratio or low spatial std → bleed-through → low score.
    # Empirical bounds: ratio > 0.15 = very bad; < 0.02 = clean.
    ratio_score = linear_score(bleedthrough_ratio, 0.02, 0.18, invert=True)
    # Spatial std < 0.3 with non-trivial ratio may indicate bleed
    spatial_score = linear_score(spatial_std, 0.15, 0.55)
    score = 0.7 * ratio_score + 0.3 * spatial_score

    normalized: dict[str, float] = {"ink_bleedthrough": round(score, 2)}
    return raw, normalized
