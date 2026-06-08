"""Integration tests: crop-level metrics flow into OCRQualityResult correctly."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np

from doc_qual import compute_doc_qual_score


def _clear_doc() -> np.ndarray:
    img = np.ones((480, 640), dtype=np.uint8) * 240
    for idx, y in enumerate(range(80, 420, 70)):
        cv2.putText(
            img,
            f"Sample line {idx + 1}",
            (60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            20,
            2,
            cv2.LINE_AA,
        )
    return img


def _blank_doc() -> np.ndarray:
    return np.ones((400, 600), dtype=np.uint8) * 250


# ── crop_analysis is populated on the result ──────────────────────────────


def test_crop_analysis_present_for_clean_doc() -> None:
    result = compute_doc_qual_score(_clear_doc(), verbose=False)
    ca = result.crop_analysis

    assert ca, "crop_analysis must be populated by the scorer"
    assert ca["crop_count"] > 0
    assert ca["valid_crop_count"] > 0
    assert ca["fallback_used"] is False
    assert isinstance(ca["crop_metrics"], list)
    assert len(ca["crop_metrics"]) == ca["valid_crop_count"]
    assert "warning" not in ca

    # Per-crop records have the documented shape.
    sample = ca["crop_metrics"][0]
    assert {"crop_id", "bbox", "blur_score", "edge_score"} <= sample.keys()
    assert len(sample["bbox"]) == 4
    assert sample["blur_score"] >= 0.0
    assert 0.0 <= sample["edge_score"] <= 1.0


def test_crop_analysis_internally_consistent_with_raw_features() -> None:
    """Document-level crop scores must match what blurriness/edges report."""
    result = compute_doc_qual_score(_clear_doc(), verbose=False)
    ca = result.crop_analysis

    # blurriness_features and edge_features pull from the same crop_metrics
    # the scorer surfaced, so the raw values must agree exactly (modulo
    # rounding inside crop_analysis).
    assert (
        abs(ca["document_blur_score"] - result.raw_features["laplacian_variance"])
        < 1e-3
    )
    assert abs(ca["document_edge_score"] - result.raw_features["edge_density"]) < 1e-5
    assert result.raw_features["blur_crop_count"] == ca["valid_crop_count"]
    assert result.raw_features["edge_crop_count"] == ca["valid_crop_count"]
    # Sanity: blur and edge agree on whether the fallback fired.
    assert (
        result.raw_features["blur_fallback_used"]
        == result.raw_features["edge_fallback_used"]
    )


# ── fallback path flows through to recommendations ────────────────────────


def test_blank_image_triggers_fallback_and_recommendation() -> None:
    result = compute_doc_qual_score(_blank_doc(), verbose=False)
    ca = result.crop_analysis

    assert ca["fallback_used"] is True
    assert ca["valid_crop_count"] == 0
    assert ca["crop_metrics"] == []
    assert "warning" in ca
    assert any(
        "fall back" in rec.lower() or "text regions" in rec.lower()
        for rec in result.recommendations
    ), result.recommendations


def test_clear_image_has_no_fallback_recommendation() -> None:
    result = compute_doc_qual_score(_clear_doc(), verbose=False)
    assert all("fall back" not in rec.lower() for rec in result.recommendations)


# ── single crop-detection call per score ──────────────────────────────────


def test_crop_detection_runs_only_once_per_score() -> None:
    """The scorer must share a single detection pass between blur and edges."""
    from doc_qual import scorer as scorer_module

    real = scorer_module.calculate_document_blur_edge_metrics
    with patch.object(
        scorer_module,
        "calculate_document_blur_edge_metrics",
        side_effect=real,
    ) as spy:
        compute_doc_qual_score(_clear_doc(), verbose=False)
    assert spy.call_count == 1, f"crop detection ran {spy.call_count} times; expected 1"


# ── to_dict serialization carries the new field ───────────────────────────


def test_to_dict_includes_crop_analysis() -> None:
    result = compute_doc_qual_score(_clear_doc(), verbose=False)
    payload = result.to_dict()
    assert "crop_analysis" in payload
    assert payload["crop_analysis"]["valid_crop_count"] > 0
    # Existing keys remain.
    assert {
        "ocr_score",
        "passed",
        "threshold",
        "feature_scores",
        "raw_features",
        "weights",
        "recommendations",
    } <= payload.keys()


# ── backward compatibility ────────────────────────────────────────────────


def test_default_construction_keeps_crop_analysis_empty() -> None:
    """Existing callers building OCRQualityResult directly remain valid."""
    from doc_qual.result import OCRQualityResult

    r = OCRQualityResult(ocr_score=80.0, passed=True, threshold=60.0)
    assert r.crop_analysis == {}
    payload = r.to_dict()
    assert payload["crop_analysis"] == {}
