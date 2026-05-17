"""Ridge response metrics for stroke-like text structure."""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def ridge_features(gray: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate text stroke structure using Hessian eigenvalue response."""

    src = gray.astype(np.float32) / 255.0
    dxx = cv2.Sobel(src, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(src, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(src, cv2.CV_32F, 1, 1, ksize=3)
    trace = dxx + dyy
    determinant_term = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy**2)
    eig1 = 0.5 * (trace + determinant_term)
    eig2 = 0.5 * (trace - determinant_term)
    response = float(np.percentile(np.maximum(np.abs(eig1), np.abs(eig2)), 95))
    score = linear_score(response, 0.015, 0.22)
    return {"ridge_response": response}, {"ridges": score}
