"""Shared utilities for loading images and reporting scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

ImageInput = Union[str, Path, np.ndarray]


def load_grayscale_image(image: ImageInput) -> np.ndarray:
    """Load a path or numpy array as an 8-bit grayscale image."""

    if isinstance(image, np.ndarray):
        arr: np.ndarray = image
    else:
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise ValueError(f"Could not read image: {path}")
        arr = loaded

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D grayscale or 3D color image")

    if arr.dtype != np.uint8:
        out = np.empty_like(arr, dtype=np.uint8)
        cv2.normalize(arr, out, 0, 255, cv2.NORM_MINMAX)
        arr = out

    return arr


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a numeric value to a closed range."""

    return float(max(lower, min(upper, value)))


def linear_score(
    value: float,
    lower: float,
    upper: float,
    *,
    invert: bool = False,
) -> float:
    """Map a value linearly onto a 0-100 scale."""

    if upper == lower:
        raise ValueError("upper and lower must differ")
    score = (value - lower) / (upper - lower) * 100.0
    if invert:
        score = 100.0 - score
    return clamp(score)


def _bar(value: float, width: int = 24) -> str:
    """Return a compact ASCII bar for terminal reports."""

    filled = int(round(clamp(value) / 100.0 * width))
    return "#" * filled + "-" * (width - filled)


def print_report(result: Any) -> None:
    """Print a readable Doc-Qual report."""

    print(f"Doc-Qual score: {result.ocr_score:.1f}/100")
    print(f"Threshold: {result.threshold:.1f} ({'PASS' if result.passed else 'FAIL'})")
    print()
    for name, score in sorted(result.feature_scores.items()):
        print(f"{name:12s} [{_bar(score)}] {score:5.1f}")
    if result.recommendations:
        print()
        print("Recommendations:")
        for item in result.recommendations:
            print(f"- {item}")
