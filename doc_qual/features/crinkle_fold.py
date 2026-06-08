"""Crinkle / fold / physical distortion detector.

Physical folds and crinkles create sharp, localised brightness discontinuities
in regions that should be flat background.  They cause local geometric
distortion that confuses OCR character segmentation.

This detector separates ink (text) regions from paper regions, then measures
unexpected high-frequency gradient energy in the paper background.  Text
regions are excluded because they legitimately contain edges and gradients.
"""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def crinkle_fold_features(
    gray: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Detect fold lines and crinkle distortion in document images.

    Strategy:
    - Separate paper background from ink/text using a large-kernel opening
      (morphological background estimation) followed by Otsu thresholding.
    - Compute the Laplacian of Gaussian (LoG) of the image as a sharpness /
      local-curvature measure.  Folds produce elongated high-curvature ridges
      even in regions that should be blank paper.
    - Measure the 95th-percentile LoG energy restricted to the paper mask
      (excluding text pixels).  High energy in the background = fold/crinkle.
    - Also measure the spatial standard deviation of background LoG to
      distinguish uniform micro-texture (noise) from localized fold ridges.

    Args:
        gray: Grayscale uint8 image.

    Returns:
        (raw_features, normalized_scores).  ``crinkle_fold`` score:
        100 = flat/clean; 0 = severely folded/crinkled.
    """
    h, w = gray.shape

    # ── Paper background mask ─────────────────────────────────────────────
    # Morphological opening with a large disk-like kernel estimates the
    # illumination background (removes text/ink strokes).
    k_size = max(int(min(h, w) * 0.04) | 1, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg_estimate = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Background pixels are those close to the estimated background value.
    diff = cv2.absdiff(gray, bg_estimate)
    _, ink_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dilate the ink mask generously to exclude all near-text border pixels
    # where the Laplacian response bleeds from ink edges into background.
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ink_mask_dilated = cv2.dilate(ink_mask, dil_k, iterations=3)
    paper_mask = cv2.bitwise_not(ink_mask_dilated)  # 255 where paper, 0 where ink

    # ── Laplacian of Gaussian on the smoothed background estimate ─────────
    # Smooth with moderate sigma (3 px) to remove noise before LoG
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.5)
    log = cv2.Laplacian(smoothed, cv2.CV_32F)
    log_abs = np.abs(log)

    # ── Background LoG statistics ─────────────────────────────────────────
    paper_pixels = log_abs[paper_mask > 0]
    if len(paper_pixels) < 100:
        # Fallback: whole-image LoG
        paper_pixels = log_abs.ravel()

    p95 = float(np.percentile(paper_pixels, 95))
    bg_log_std = float(np.std(paper_pixels))

    # ── Fold ridge detection: look for connected high-LoG regions in paper ─
    # A fold produces a contiguous high-curvature band; noise is diffuse.
    threshold_val = max(p95 * 0.5, 10.0)
    high_curv = (log_abs > threshold_val).astype(np.uint8) * 255
    high_curv_in_paper = cv2.bitwise_and(high_curv, paper_mask)

    # Count ratio of high-curvature paper pixels
    paper_px_count = max(int(np.sum(paper_mask > 0)), 1)
    fold_pixel_ratio = float(np.sum(high_curv_in_paper > 0)) / paper_px_count

    raw: dict[str, float] = {
        "fold_pixel_ratio": round(fold_pixel_ratio, 6),
        "bg_log_p95": round(p95, 4),
        "bg_log_std": round(bg_log_std, 4),
    }

    # Scoring: fold_pixel_ratio 0–0.10 = acceptable; > 0.40 = severe fold
    ratio_score = linear_score(fold_pixel_ratio, 0.0, 0.40, invert=True)
    # p95 LoG: < 10 = clean paper, > 80 = crinkled
    p95_score = linear_score(p95, 5.0, 80.0, invert=True)
    score = 0.6 * ratio_score + 0.4 * p95_score

    normalized: dict[str, float] = {"crinkle_fold": round(score, 2)}
    return raw, normalized
