import cv2
import numpy as np

from doc_qual import OCRQualityResult, compute_doc_qual_score


def make_clean_doc() -> np.ndarray:
    img = np.ones((300, 500), dtype=np.uint8) * 240
    for y in range(40, 260, 25):
        img[y : y + 6, 50:450] = 25
    return img


def test_clean_image_scores_high() -> None:
    result = compute_doc_qual_score(make_clean_doc(), verbose=False)
    assert isinstance(result, OCRQualityResult)
    assert result.ocr_score >= 60


def test_blurry_image_scores_lower() -> None:
    clean = make_clean_doc()
    blurry = cv2.GaussianBlur(clean, (21, 21), 0)
    r_clean = compute_doc_qual_score(clean, verbose=False)
    r_blurry = compute_doc_qual_score(blurry, verbose=False)
    assert r_blurry.ocr_score < r_clean.ocr_score


def test_returns_score_in_range() -> None:
    result = compute_doc_qual_score(make_clean_doc(), verbose=False)
    assert 0 <= result.ocr_score <= 100


def test_threshold_controls_pass_flag() -> None:
    result = compute_doc_qual_score(make_clean_doc(), threshold=99, verbose=False)
    assert result.threshold == 99
    assert result.passed is (result.ocr_score >= 99)
