"""Composite Doc-Qual scoring."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from doc_qual.features import (
    blurriness_features,
    brightness_features,
    edge_features,
    noise_features,
    ridge_features,
    skew_features,
)
from doc_qual.result import OCRQualityResult
from doc_qual.utils import ImageInput, clamp, load_grayscale_image, print_report

FeatureExtractor = Callable[[np.ndarray], tuple[dict[str, float], dict[str, float]]]

DEFAULT_WEIGHTS: dict[str, float] = {
    "sharpness": 0.25,
    "noise": 0.18,
    "edges": 0.17,
    "skew": 0.15,
    "brightness": 0.15,
    "ridges": 0.10,
}

FEATURE_EXTRACTORS: tuple[FeatureExtractor, ...] = (
    blurriness_features,
    noise_features,
    edge_features,
    skew_features,
    brightness_features,
    ridge_features,
)


def compute_doc_qual_score(
    image: ImageInput,
    *,
    threshold: float = 60.0,
    weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> OCRQualityResult:
    """Compute a no-reference OCR readiness score for a document image.

    Args:
        image: Image path or numpy array. Color arrays are converted to grayscale.
        threshold: Minimum passing composite score.
        weights: Optional feature weights keyed by normalized feature name.
        verbose: Print a text report when true.

    Returns:
        OCRQualityResult with composite score, feature scores, raw measurements,
        weights, and recommendations.
    """

    gray = load_grayscale_image(image)
    active_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        active_weights.update(weights)

    raw_features: dict[str, float] = {}
    feature_scores: dict[str, float] = {}
    for extractor in FEATURE_EXTRACTORS:
        raw, normalized = extractor(gray)
        raw_features.update(raw)
        feature_scores.update(normalized)

    total_weight = sum(active_weights.get(name, 0.0) for name in feature_scores)
    if total_weight <= 0:
        raise ValueError("At least one feature weight must be positive")

    weighted_score = sum(
        clamp(score) * active_weights.get(name, 0.0)
        for name, score in feature_scores.items()
    )
    ocr_score = clamp(weighted_score / total_weight)
    result = OCRQualityResult(
        ocr_score=ocr_score,
        passed=ocr_score >= threshold,
        threshold=threshold,
        feature_scores=feature_scores,
        raw_features=raw_features,
        weights={name: active_weights[name] for name in feature_scores},
        recommendations=_recommend(feature_scores),
    )

    if verbose:
        print_report(result)

    return result


def _recommend(feature_scores: dict[str, float]) -> list[str]:
    recommendations: list[str] = []
    if feature_scores.get("sharpness", 100.0) < 55.0:
        recommendations.append("Rescan or sharpen the image to improve text edges.")
    if feature_scores.get("noise", 100.0) < 55.0:
        recommendations.append("Denoise the image or use a cleaner scan source.")
    if feature_scores.get("skew", 100.0) < 55.0:
        recommendations.append("Deskew the page before OCR.")
    if feature_scores.get("brightness", 100.0) < 55.0:
        recommendations.append("Improve exposure or contrast before OCR.")
    if feature_scores.get("edges", 100.0) < 45.0:
        recommendations.append("Check that the document contains readable text.")
    return recommendations
