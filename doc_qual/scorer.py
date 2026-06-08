"""Composite Doc-Qual scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from doc_qual.features import (
    blurriness_features,
    brightness_features,
    brisque_like_features,
    crinkle_fold_features,
    edge_features,
    ink_bleedthrough_features,
    local_contrast_features,
    noise_features,
    ridge_features,
    shadow_gradient_features,
    skew_features,
)
from doc_qual.features.text_crops import calculate_document_blur_edge_metrics
from doc_qual.result import OCRQualityResult
from doc_qual.utils import ImageInput, clamp, load_grayscale_image, print_report

FeatureExtractor = Callable[[np.ndarray], tuple[dict[str, float], dict[str, float]]]

DEFAULT_WEIGHTS: dict[str, float] = {
    "sharpness": 0.22,
    "noise": 0.16,
    "edges": 0.15,
    "skew": 0.13,
    "brightness": 0.13,
    "ridges": 0.08,
    "ink_bleedthrough": 0.04,
    "shadow_gradient": 0.04,
    "local_contrast": 0.03,
    "crinkle_fold": 0.02,
    "brisque": 0.00,  # off by default until empirically calibrated; enable via weights=
}

# Crop-aware extractors consume a precomputed crop-detection result; the rest
# operate on the full image only and need no extra context.
_CROP_AWARE_EXTRACTORS: tuple[
    Callable[..., tuple[dict[str, float], dict[str, float]]], ...
] = (
    blurriness_features,
    edge_features,
)
_INDEPENDENT_EXTRACTORS: tuple[FeatureExtractor, ...] = (
    noise_features,
    skew_features,
    brightness_features,
    ridge_features,
    ink_bleedthrough_features,
    shadow_gradient_features,
    local_contrast_features,
    crinkle_fold_features,
    brisque_like_features,
)

# Back-compat alias — historical tuple of every extractor used by the scorer.
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
    engine: str | None = None,
    verbose: bool = True,
) -> OCRQualityResult:
    """Compute a no-reference OCR readiness score for a document image.

    Args:
        image: Image path or numpy array. Color arrays are converted to grayscale.
        threshold: Minimum passing composite score.
        weights: Optional feature weight overrides (merged on top of the
            profile or default weights).
        engine: OCR engine profile name — one of ``'tesseract'``,
            ``'textract'``, ``'azure'``, ``'paddleocr'``, or a path to a
            custom JSON weight file.  When set, the engine profile is used
            as the base weight set before ``weights`` overrides are applied.
        verbose: Print a text report when true.

    Returns:
        OCRQualityResult with composite score, feature scores, raw measurements,
        weights, recommendations and a ``crop_analysis`` summary describing
        the text crops used to compute the blur and edge metrics.
    """
    gray = load_grayscale_image(image)

    # Build active weights: default → engine profile → explicit overrides
    active_weights = dict(DEFAULT_WEIGHTS)
    if engine is not None:
        from doc_qual.profiles import load_engine_profile

        profile = load_engine_profile(engine)
        active_weights.update(profile)
    if weights:
        active_weights.update(weights)

    # Run crop detection once and share it between the blur and edge features.
    crop_metrics = calculate_document_blur_edge_metrics(gray)

    raw_features: dict[str, float] = {}
    feature_scores: dict[str, float] = {}

    for extractor in _CROP_AWARE_EXTRACTORS:
        raw, normalized = extractor(gray, crop_metrics=crop_metrics)
        raw_features.update(raw)
        feature_scores.update(normalized)

    for extractor in _INDEPENDENT_EXTRACTORS:
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

    crop_analysis = _build_crop_analysis(crop_metrics)
    recommendations = _recommend(feature_scores, crop_metrics)

    result = OCRQualityResult(
        ocr_score=ocr_score,
        passed=ocr_score >= threshold,
        threshold=threshold,
        feature_scores=feature_scores,
        raw_features=raw_features,
        weights={name: active_weights.get(name, 0.0) for name in feature_scores},
        recommendations=recommendations,
        crop_analysis=crop_analysis,
        engine=engine,
    )

    if verbose:
        print_report(result)

    return result


def _build_crop_analysis(crop_metrics: dict[str, Any]) -> dict[str, Any]:
    """Project the internal crop-metric dict into a stable, serialisable shape."""
    analysis: dict[str, Any] = {
        "crop_count": int(crop_metrics["crop_count"]),
        "valid_crop_count": int(crop_metrics["valid_crop_count"]),
        "fallback_used": bool(crop_metrics["fallback_used"]),
        "document_blur_score": round(float(crop_metrics["document_blur_score"]), 4),
        "document_edge_score": round(float(crop_metrics["document_edge_score"]), 6),
        "crop_metrics": list(crop_metrics.get("crop_metrics", [])),
    }
    if "warning" in crop_metrics:
        analysis["warning"] = crop_metrics["warning"]
    return analysis


def _recommend(
    feature_scores: dict[str, float],
    crop_metrics: dict[str, Any] | None = None,
) -> list[str]:
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
    if feature_scores.get("ink_bleedthrough", 100.0) < 50.0:
        recommendations.append(
            "Ink bleed-through detected — scan single-sided or use a backing sheet."
        )
    if feature_scores.get("shadow_gradient", 100.0) < 50.0:
        recommendations.append(
            "Uneven lighting detected — use a flatbed scanner or improve lighting."
        )
    if feature_scores.get("local_contrast", 100.0) < 50.0:
        recommendations.append(
            "Low local contrast — increase scanner exposure or use image enhancement."
        )
    if feature_scores.get("crinkle_fold", 100.0) < 50.0:
        recommendations.append(
            "Fold lines or crinkle detected — flatten the document before scanning."
        )
    if crop_metrics is not None and crop_metrics.get("fallback_used"):
        recommendations.append(
            "Could not isolate text regions; blur and edge metrics fall back "
            "to full-image values. Verify the page contains readable text."
        )
    return recommendations
