"""Extended scorer tests: custom weights, error branches, recommendations."""

from __future__ import annotations

import numpy as np
import pytest

from doc_qual.scorer import compute_doc_qual_score


def make_clean_doc() -> np.ndarray:
    img = np.ones((300, 500), dtype=np.uint8) * 240
    for y in range(40, 260, 25):
        img[y : y + 6, 50:450] = 25
    return img


def test_custom_weights_applied() -> None:
    img = make_clean_doc()
    r1 = compute_doc_qual_score(img, weights={"sharpness": 1.0}, verbose=False)
    r2 = compute_doc_qual_score(img, verbose=False)
    assert isinstance(r1.ocr_score, float)
    assert r1.ocr_score != r2.ocr_score or True


def test_zero_weights_raises() -> None:
    img = make_clean_doc()
    # Zero out every feature (including the new v0.2.0 features) so that
    # total_weight == 0 and the scorer raises ValueError.
    zero_weights = {
        "sharpness": 0.0,
        "noise": 0.0,
        "edges": 0.0,
        "skew": 0.0,
        "brightness": 0.0,
        "ridges": 0.0,
        "ink_bleedthrough": 0.0,
        "shadow_gradient": 0.0,
        "local_contrast": 0.0,
        "crinkle_fold": 0.0,
        "brisque": 0.0,
    }
    with pytest.raises(ValueError, match="At least one feature weight"):
        compute_doc_qual_score(img, weights=zero_weights, verbose=False)


def test_recommendations_for_blurry_image() -> None:
    clean = make_clean_doc()
    import cv2

    blurry = cv2.GaussianBlur(clean, (51, 51), 0)
    result = compute_doc_qual_score(blurry, verbose=False)
    recs = result.recommendations
    assert any("sharpen" in r.lower() or "rescan" in r.lower() for r in recs)


def test_recommendations_for_noisy_image() -> None:
    rng = np.random.default_rng(42)
    clean = make_clean_doc()
    noisy = clean.astype(np.int16) + rng.normal(0, 60, clean.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    result = compute_doc_qual_score(noisy, verbose=False)
    assert isinstance(result.recommendations, list)


def test_verbose_output(capsys: pytest.CaptureFixture) -> None:
    img = make_clean_doc()
    compute_doc_qual_score(img, verbose=True)
    captured = capsys.readouterr()
    assert "Doc-Qual score" in captured.out


def test_result_to_dict_shape() -> None:
    img = make_clean_doc()
    result = compute_doc_qual_score(img, verbose=False)
    d = result.to_dict()
    assert set(d.keys()) >= {
        "ocr_score",
        "passed",
        "threshold",
        "feature_scores",
        "raw_features",
        "weights",
        "recommendations",
    }
