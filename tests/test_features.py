import cv2
import numpy as np

from doc_qual.features import (
    blurriness_features,
    brightness_features,
    edge_features,
    noise_features,
    ridge_features,
    skew_features,
)


def make_clean_doc() -> np.ndarray:
    img = np.ones((240, 360), dtype=np.uint8) * 235
    for y in range(35, 210, 24):
        img[y : y + 5, 40:320] = 30
    return img


def test_blur_feature_decreases_after_blur() -> None:
    clean = make_clean_doc()
    blurry = cv2.GaussianBlur(clean, (19, 19), 0)
    _, clean_scores = blurriness_features(clean)
    _, blurry_scores = blurriness_features(blurry)
    assert blurry_scores["sharpness"] < clean_scores["sharpness"]


def test_feature_scores_are_in_range() -> None:
    img = make_clean_doc()
    extractors = [
        blurriness_features,
        brightness_features,
        edge_features,
        noise_features,
        ridge_features,
        skew_features,
    ]
    for extractor in extractors:
        _, scores = extractor(img)
        assert scores
        assert all(0 <= value <= 100 for value in scores.values())


def test_noise_feature_decreases_with_synthetic_noise() -> None:
    rng = np.random.default_rng(7)
    clean = make_clean_doc()
    noisy = clean.astype(np.int16) + rng.normal(0, 35, clean.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    _, clean_scores = noise_features(clean)
    _, noisy_scores = noise_features(noisy)
    assert noisy_scores["noise"] < clean_scores["noise"]
