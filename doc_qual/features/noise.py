"""Noise estimation for document images."""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def noise_features(gray: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate high-frequency noise with a Gaussian residual."""

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(np.std(residual))
    score = linear_score(noise_std, 4.0, 28.0, invert=True)
    return {"noise_std": noise_std}, {"noise": score}
