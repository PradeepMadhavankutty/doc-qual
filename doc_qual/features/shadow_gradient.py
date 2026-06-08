"""Shadow / uneven-lighting detector.

Phone scans and flatbed scans with poor lamp uniformity exhibit brightness
gradients — one corner is dark while the opposite is bright.  This is
distinct from global under-exposure (handled by brightness.py) and is
measured by the *spatial* variance of local brightness across a coarse grid.
"""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def shadow_gradient_features(
    gray: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Detect uneven lighting / shadow gradients across the page.

    Strategy:
    - Divide the image into an 8 × 8 grid of cells.
    - Compute the mean brightness of each cell.
    - High standard deviation across cells indicates a shadow or uneven light.
    - Also measure the range (max − min cell mean) as a complementary signal.

    Args:
        gray: Grayscale uint8 image.

    Returns:
        (raw_features, normalized_scores).  ``shadow_gradient`` score:
        100 = perfectly uniform; 0 = severe gradient.
    """
    h, w = gray.shape

    grid_r, grid_c = 8, 8
    cell_h = max(1, h // grid_r)
    cell_w = max(1, w // grid_c)

    # Use a very large Gaussian blur (sigma ≈ 5% of image width) to suppress
    # text and capture only the illumination envelope of the page.
    k = max(int(min(h, w) * 0.05) | 1, 21)  # odd kernel, at least 21 px
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    cell_means: list[float] = []
    for r in range(grid_r):
        for c in range(grid_c):
            y0, y1 = r * cell_h, min((r + 1) * cell_h, h)
            x0, x1 = c * cell_w, min((c + 1) * cell_w, w)
            cell = blurred[y0:y1, x0:x1]
            if cell.size > 0:
                # 90th-percentile approximates the paper white-point per cell,
                # robustly excluding dark ink pixels regardless of text density.
                cell_means.append(float(np.percentile(cell, 90)))

    if not cell_means:
        return {"shadow_brightness_std": 0.0, "shadow_range": 0.0}, {
            "shadow_gradient": 100.0
        }

    arr = np.array(cell_means, dtype=np.float32)
    brightness_std = float(np.std(arr))
    brightness_range = float(arr.max() - arr.min())

    raw: dict[str, float] = {
        "shadow_brightness_std": round(brightness_std, 4),
        "shadow_range": round(brightness_range, 4),
    }

    # Std of cell means: 0–8 = fine, 25+ = severe.
    # Std bounds: text lines naturally create ~15–25 std across cells;
    # genuine shadows/gradients push this above 35.
    std_score = linear_score(brightness_std, 2.0, 45.0, invert=True)
    # Range: text contrast gives 30–60; full shadow gradient reaches 150+.
    range_score = linear_score(brightness_range, 15.0, 150.0, invert=True)
    score = 0.6 * std_score + 0.4 * range_score

    normalized: dict[str, float] = {"shadow_gradient": round(score, 2)}
    return raw, normalized
