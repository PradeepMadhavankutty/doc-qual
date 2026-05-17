"""Tests for doc_qual.utils covering file-loading and helper functions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_qual.result import OCRQualityResult
from doc_qual.utils import clamp, linear_score, load_grayscale_image, print_report


def _write_gray_png(path: Path, arr: np.ndarray) -> Path:
    cv2.imwrite(str(path), arr)
    return path


def _make_gray() -> np.ndarray:
    img = np.ones((100, 100), dtype=np.uint8) * 200
    img[10:20, 10:90] = 50
    return img


def test_load_grayscale_from_ndarray_passthrough() -> None:
    arr = _make_gray()
    out = load_grayscale_image(arr)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8


def test_load_grayscale_from_path(tmp_path: Path) -> None:
    img = _make_gray()
    p = _write_gray_png(tmp_path / "test.png", img)
    out = load_grayscale_image(p)
    assert out.ndim == 2
    assert out.dtype == np.uint8


def test_load_grayscale_from_str_path(tmp_path: Path) -> None:
    img = _make_gray()
    p = _write_gray_png(tmp_path / "test.png", img)
    out = load_grayscale_image(str(p))
    assert out.ndim == 2


def test_load_grayscale_converts_color(tmp_path: Path) -> None:
    color = np.zeros((80, 80, 3), dtype=np.uint8)
    color[:, :, 0] = 200
    p = tmp_path / "color.png"
    cv2.imwrite(str(p), color)
    out = load_grayscale_image(p)
    assert out.ndim == 2


def test_load_grayscale_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_grayscale_image("/nonexistent/path/img.png")


def test_load_grayscale_rejects_4d_array() -> None:
    arr = np.zeros((10, 10, 3, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D grayscale or 3D color"):
        load_grayscale_image(arr)


def test_clamp_within_range() -> None:
    assert clamp(50.0) == 50.0


def test_clamp_below_lower() -> None:
    assert clamp(-10.0) == 0.0


def test_clamp_above_upper() -> None:
    assert clamp(150.0) == 100.0


def test_clamp_custom_bounds() -> None:
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(11.0, 0.0, 10.0) == 10.0


def test_linear_score_equal_bounds_raises() -> None:
    with pytest.raises(ValueError, match="upper and lower must differ"):
        linear_score(5.0, 5.0, 5.0)


def test_linear_score_normal() -> None:
    score = linear_score(5.0, 0.0, 10.0)
    assert score == pytest.approx(50.0)


def test_linear_score_inverted() -> None:
    score = linear_score(5.0, 0.0, 10.0, invert=True)
    assert score == pytest.approx(50.0)


def test_print_report_runs(capsys: pytest.CaptureFixture) -> None:
    result = OCRQualityResult(
        ocr_score=75.0,
        passed=True,
        threshold=60.0,
        feature_scores={"sharpness": 80.0, "noise": 70.0},
        raw_features={},
        weights={"sharpness": 0.5, "noise": 0.5},
        recommendations=["Rescan or sharpen the image to improve text edges."],
    )
    print_report(result)
    captured = capsys.readouterr()
    assert "75.0/100" in captured.out
    assert "PASS" in captured.out
    assert "Rescan" in captured.out
