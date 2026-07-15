"""Skew estimation using Hough line angles."""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score


def skew_features(gray: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate document skew in degrees and score deskew readiness."""

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(30, gray.shape[1] // 8),
        maxLineGap=12,
    )
    if lines is None:
        return {"skew_angle": 0.0}, {"skew": 80.0}

    angles: list[float] = []
    for row in lines.reshape(-1, 4):
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if -45.0 <= angle <= 45.0:
            angles.append(float(angle))

    skew_angle = float(np.median(angles)) if angles else 0.0
    score = linear_score(abs(skew_angle), 0.5, 10.0, invert=True)
    return {"skew_angle": skew_angle}, {"skew": score}
