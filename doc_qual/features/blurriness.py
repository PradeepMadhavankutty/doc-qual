"""Blur estimation using Laplacian variance."""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def blurriness_features(gray: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Compute blur-related raw values and normalized scores."""

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = linear_score(variance, 35.0, 650.0)
    return {"laplacian_variance": variance}, {"sharpness": score}
