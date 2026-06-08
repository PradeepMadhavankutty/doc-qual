"""Unit tests for crop-level blur and edge metric helpers."""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.features import (
    blurriness_features,
    calculate_crop_blur_score,
    calculate_crop_edge_score,
    calculate_document_blur_edge_metrics,
    detect_text_crops,
    edge_features,
)

# ── synthetic image factories ─────────────────────────────────────────────


def _multi_block_doc() -> np.ndarray:
    """A document with several visually distinct text blocks."""
    img = np.ones((600, 800), dtype=np.uint8) * 245
    cv2.putText(
        img,
        "INVOICE 2026",
        (60, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        20,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Customer: Acme Corporation",
        (60, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        25,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Amount Due: $1432.50",
        (60, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        25,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Payment due in 30 days",
        (60, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        25,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Thank you for your business",
        (60, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        25,
        2,
        cv2.LINE_AA,
    )
    return img


def _clear_doc() -> np.ndarray:
    """Crisp document with several text lines."""
    img = np.ones((480, 640), dtype=np.uint8) * 240
    for idx, y in enumerate(range(80, 420, 70)):
        cv2.putText(
            img,
            f"Sample line {idx + 1} of text",
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


def _noisy_doc() -> np.ndarray:
    rng = np.random.default_rng(11)
    base = _clear_doc().astype(np.int16)
    noisy = base + rng.normal(0, 18, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ── detect_text_crops ─────────────────────────────────────────────────────


def test_detect_text_crops_finds_multiple_regions() -> None:
    img = _multi_block_doc()
    crops = detect_text_crops(img)
    assert len(crops) >= 3
    for crop in crops:
        bbox = crop["bbox"]
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert w >= 20 and h >= 10
        assert crop["crop"].shape == (h, w)


def test_detect_text_crops_blank_image_returns_empty() -> None:
    crops = detect_text_crops(_blank_doc())
    assert crops == []


def test_detect_text_crops_accepts_color_input() -> None:
    gray = _clear_doc()
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    crops_gray = detect_text_crops(gray)
    crops_color = detect_text_crops(color)
    assert len(crops_color) == len(crops_gray)


# ── per-crop helpers ──────────────────────────────────────────────────────


def test_crop_blur_decreases_with_blur() -> None:
    img = _clear_doc()
    crops = detect_text_crops(img)
    assert crops, "expected at least one crop in a clear document"
    crop = crops[0]["crop"]
    blurred_crop = cv2.GaussianBlur(crop, (15, 15), 0)
    assert calculate_crop_blur_score(blurred_crop) < calculate_crop_blur_score(crop)


def test_crop_edge_score_is_a_density() -> None:
    img = _clear_doc()
    crops = detect_text_crops(img)
    assert crops
    score = calculate_crop_edge_score(crops[0]["crop"])
    assert 0.0 <= score <= 1.0


# ── document-level aggregation ────────────────────────────────────────────


def test_document_metrics_clear_image() -> None:
    result = calculate_document_blur_edge_metrics(_clear_doc())
    assert result["valid_crop_count"] > 0
    assert "document_blur_score" in result
    assert "document_edge_score" in result
    assert isinstance(result["crop_metrics"], list)
    assert result["fallback_used"] is False
    assert "warning" not in result

    for entry in result["crop_metrics"]:
        assert {"crop_id", "bbox", "blur_score", "edge_score"} <= entry.keys()
        assert len(entry["bbox"]) == 4
        assert entry["blur_score"] >= 0.0
        assert 0.0 <= entry["edge_score"] <= 1.0


def test_document_metrics_multi_block_image() -> None:
    result = calculate_document_blur_edge_metrics(_multi_block_doc())
    assert result["valid_crop_count"] >= 3
    assert result["fallback_used"] is False
    mean_blur = sum(c["blur_score"] for c in result["crop_metrics"]) / len(
        result["crop_metrics"]
    )
    assert (
        result["document_blur_score"] == round(mean_blur, 4)
        or abs(result["document_blur_score"] - mean_blur) < 1e-3
    )


def test_document_metrics_blur_decreases_with_blur() -> None:
    clear = _clear_doc()
    blurry = cv2.GaussianBlur(clear, (21, 21), 0)
    r_clear = calculate_document_blur_edge_metrics(clear)
    r_blur = calculate_document_blur_edge_metrics(blurry)
    assert r_blur["document_blur_score"] < r_clear["document_blur_score"]


def test_document_metrics_blank_image_falls_back() -> None:
    result = calculate_document_blur_edge_metrics(_blank_doc())
    assert result["valid_crop_count"] == 0
    assert result["fallback_used"] is True
    assert "warning" in result
    assert "document_blur_score" in result
    assert "document_edge_score" in result


def test_document_metrics_noisy_image_still_returns_crops() -> None:
    result = calculate_document_blur_edge_metrics(_noisy_doc())
    # Mild noise should not destroy the text segmentation.
    assert result["valid_crop_count"] > 0
    assert result["document_blur_score"] > 0.0


# ── public feature extractors keep working ────────────────────────────────


def test_blurriness_features_exposes_crop_metadata() -> None:
    img = _clear_doc()
    raw, scores = blurriness_features(img)
    assert "laplacian_variance" in raw
    assert "blur_crop_count" in raw
    assert raw["blur_crop_count"] > 0
    assert raw["blur_fallback_used"] == 0.0
    assert 0.0 <= scores["sharpness"] <= 100.0


def test_edge_features_exposes_crop_metadata() -> None:
    img = _clear_doc()
    raw, scores = edge_features(img)
    assert "edge_density" in raw
    assert "edge_crop_count" in raw
    assert raw["edge_crop_count"] > 0
    assert raw["edge_fallback_used"] == 0.0
    assert 0.0 <= scores["edges"] <= 100.0


def test_blurriness_features_falls_back_on_blank() -> None:
    raw, scores = blurriness_features(_blank_doc())
    assert raw["blur_fallback_used"] == 1.0
    assert raw["blur_crop_count"] == 0.0
    assert 0.0 <= scores["sharpness"] <= 100.0


def test_edge_features_falls_back_on_blank() -> None:
    raw, scores = edge_features(_blank_doc())
    assert raw["edge_fallback_used"] == 1.0
    assert raw["edge_crop_count"] == 0.0
    assert 0.0 <= scores["edges"] <= 100.0
