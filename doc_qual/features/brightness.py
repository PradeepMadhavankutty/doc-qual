"""Brightness and contrast scoring."""

from __future__ import annotations

import numpy as np

from doc_qual.utils import clamp, linear_score


def brightness_features(gray: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Measure exposure and contrast for OCR readiness."""

    mean = float(np.mean(gray))
    std = float(np.std(gray))
    exposure_penalty = abs(mean - 185.0) / 185.0 * 100.0
    exposure_score = clamp(100.0 - exposure_penalty)
    contrast_score = linear_score(std, 25.0, 85.0)
    score = exposure_score * 0.45 + contrast_score * 0.55
    return {"brightness_mean": mean, "brightness_std": std}, {"brightness": score}
