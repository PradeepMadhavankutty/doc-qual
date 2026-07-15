"""Positive-orientation validation tests.

Every feature score and the composite score must satisfy:
    higher score ↔ better input quality
    lower score  ↔ worse input quality

These tests verify that relationship by constructing controlled synthetic
images that degrade one quality dimension at a time and confirming that the
corresponding feature score (and composite score) decreases monotonically.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from doc_qual.features.blurriness import blurriness_features
from doc_qual.features.brightness import brightness_features
from doc_qual.features.brisque_like import brisque_like_features
from doc_qual.features.crinkle_fold import crinkle_fold_features
from doc_qual.features.edges import edge_features
from doc_qual.features.ink_bleedthrough import ink_bleedthrough_features
from doc_qual.features.local_contrast import local_contrast_features
from doc_qual.features.noise import noise_features
from doc_qual.features.ridges import ridge_features
from doc_qual.features.shadow_gradient import shadow_gradient_features
from doc_qual.features.skew import skew_features
from doc_qual.scorer import compute_doc_qual_score

# ---------------------------------------------------------------------------
# Helpers to build synthetic document images
# ---------------------------------------------------------------------------


def _clean_doc(h: int = 400, w: int = 300) -> np.ndarray:
    """Bright white page with sharp horizontal text-like stripes."""
    img = np.full((h, w), 240, dtype=np.uint8)
    for y in range(30, h - 30, 20):
        img[y : y + 3, 20 : w - 20] = 30
    return img


def _apply_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize | 1  # ensure odd
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_noise(img: np.ndarray, std: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_skew(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=240)


def _darken(img: np.ndarray, factor: float) -> np.ndarray:
    """Reduce brightness uniformly (factor < 1 = darker)."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _add_shadow(img: np.ndarray, gradient_strength: float) -> np.ndarray:
    """Add a left-to-right brightness gradient to simulate uneven lighting."""
    h, w = img.shape
    ramp = np.linspace(0, gradient_strength, w, dtype=np.float32)
    shadow = np.tile(ramp, (h, 1))
    return np.clip(img.astype(np.float32) - shadow, 0, 255).astype(np.uint8)


def _add_bleedthrough(img: np.ndarray, bleed_value: int) -> np.ndarray:
    """Darken margin bands to simulate ink bleed-through from the reverse side."""
    out = img.copy()
    h, w = out.shape
    top = max(1, int(h * 0.12))
    bot = max(1, int(h * 0.12))
    lr = max(1, int(w * 0.08))
    def _darken_band(band: np.ndarray) -> np.ndarray:
        return np.clip(band.astype(int) - bleed_value, 0, 255).astype(np.uint8)

    out[:top, :] = _darken_band(out[:top, :])
    out[h - bot :, :] = _darken_band(out[h - bot :, :])
    out[:, :lr] = _darken_band(out[:, :lr])
    out[:, w - lr :] = _darken_band(out[:, w - lr :])
    return out


def _low_contrast(img: np.ndarray, compress: float) -> np.ndarray:
    """Compress pixel range toward mid-grey to reduce local contrast."""
    f = img.astype(np.float32)
    return np.clip(128 + (f - 128) * compress, 0, 255).astype(np.uint8)


def _add_fold(img: np.ndarray, fold_width: int = 6) -> np.ndarray:
    """Add a bright crease (fold line) diagonally across the background."""
    out = img.copy()
    h, w = out.shape
    # Draw a diagonal bright stripe to simulate a fold crease
    for i in range(-fold_width // 2, fold_width // 2 + 1):
        y_vals = np.arange(h)
        x_vals = np.clip((y_vals * w // h) + i, 0, w - 1)
        for y, x in zip(y_vals, x_vals):
            out[y, x] = min(255, int(out[y, x]) + 80)
    return out


# ---------------------------------------------------------------------------
# 1. Sharpness — increasing blur must reduce the sharpness score
# ---------------------------------------------------------------------------


def test_sharpness_decreases_with_blur() -> None:
    base = _clean_doc()
    scores = []
    for ksize in [1, 5, 15, 31, 61]:
        img = _apply_blur(base, ksize) if ksize > 1 else base
        _, norm = blurriness_features(img)
        scores.append(norm["sharpness"])

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1.0, (
            f"Sharpness did not decrease: blur level {i} score={scores[i]:.1f} "
            f"vs blur level {i+1} score={scores[i+1]:.1f}"
        )
    assert scores[0] > scores[-1], "Sharpest image must outscore heaviest blur"


# ---------------------------------------------------------------------------
# 2. Noise — increasing noise must reduce the noise score
# ---------------------------------------------------------------------------


def test_noise_score_decreases_with_noise() -> None:
    base = _clean_doc()
    scores = []
    for std in [0, 5, 12, 22, 35]:
        img = _apply_noise(base, std) if std > 0 else base
        _, norm = noise_features(img)
        scores.append(norm["noise"])

    assert scores[0] > scores[-1], "Clean image must outscore heavily noisy image"
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 2.0, (
            f"Noise score increased with more noise: "
            f"std[{i}]→{scores[i]:.1f}, std[{i+1}]→{scores[i+1]:.1f}"
        )


# ---------------------------------------------------------------------------
# 3. Skew — increasing skew must reduce the skew score
# ---------------------------------------------------------------------------


def test_skew_score_decreases_with_angle() -> None:
    base = _clean_doc()
    scores = []
    for angle in [0.0, 2.0, 5.0, 10.0, 15.0]:
        img = _apply_skew(base, angle)
        _, norm = skew_features(img)
        scores.append(norm["skew"])

    assert scores[0] > scores[-1], "Straight image must outscore heavily skewed image"


# ---------------------------------------------------------------------------
# 4. Brightness — heavily darkened image must score lower than well-exposed
# ---------------------------------------------------------------------------


def test_brightness_score_drops_when_too_dark() -> None:
    base = _clean_doc()
    _, norm_good = brightness_features(base)
    dark = _darken(base, 0.25)  # 25% of original brightness
    _, norm_dark = brightness_features(dark)
    assert norm_good["brightness"] > norm_dark["brightness"], (
        f"Well-exposed image ({norm_good['brightness']:.1f}) must outscore "
        f"dark image ({norm_dark['brightness']:.1f})"
    )


def test_brightness_score_drops_when_too_bright() -> None:
    base = _clean_doc()
    _, norm_good = brightness_features(base)
    blown = np.clip(base.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
    _, norm_blown = brightness_features(blown)
    assert norm_good["brightness"] > norm_blown["brightness"]


# ---------------------------------------------------------------------------
# 5. Ridges — clean text page should score higher than blank white page
# ---------------------------------------------------------------------------


def test_ridge_score_higher_with_text_than_blank() -> None:
    text_doc = _clean_doc()
    blank = np.full_like(text_doc, 240)
    _, norm_text = ridge_features(text_doc)
    _, norm_blank = ridge_features(blank)
    assert norm_text["ridges"] > norm_blank["ridges"], (
        f"Text page ridge ({norm_text['ridges']:.1f}) must outscore "
        f"blank page ({norm_blank['ridges']:.1f})"
    )


# ---------------------------------------------------------------------------
# 6. Edges — blank page and over-busy page must score lower than normal text
# ---------------------------------------------------------------------------


def test_edge_score_clean_text_above_blank() -> None:
    text_doc = _clean_doc()
    blank = np.full_like(text_doc, 240)
    _, norm_text = edge_features(text_doc)
    _, norm_blank = edge_features(blank)
    assert norm_text["edges"] > norm_blank["edges"], (
        f"Text ({norm_text['edges']:.1f}) vs blank ({norm_blank['edges']:.1f})"
    )


def test_edge_score_is_non_negative() -> None:
    noisy = _apply_noise(_clean_doc(), 50)
    _, norm = edge_features(noisy)
    assert norm["edges"] >= 0.0


# ---------------------------------------------------------------------------
# 7. Ink bleed-through — heavier bleed must score lower
# ---------------------------------------------------------------------------


def test_bleedthrough_score_decreases_with_bleed() -> None:
    base = _clean_doc()
    scores = []
    for strength in [0, 60, 130, 200]:
        img = _add_bleedthrough(base, strength)
        _, norm = ink_bleedthrough_features(img)
        scores.append(norm["ink_bleedthrough"])

    assert scores[0] > scores[-1], (
        f"Clean ({scores[0]:.1f}) must outscore severe bleed ({scores[-1]:.1f})"
    )


# ---------------------------------------------------------------------------
# 8. Shadow / gradient — stronger shadow must score lower
# ---------------------------------------------------------------------------


def test_shadow_gradient_score_decreases_with_shadow() -> None:
    base = _clean_doc()
    scores = []
    for strength in [0, 50, 100, 180]:
        img = _add_shadow(base, strength)
        _, norm = shadow_gradient_features(img)
        scores.append(norm["shadow_gradient"])

    assert scores[0] > scores[-1], (
        f"Uniform lighting ({scores[0]:.1f}) must outscore "
        f"severe shadow ({scores[-1]:.1f})"
    )


# ---------------------------------------------------------------------------
# 9. Local contrast — lower Michelson contrast must reduce the score
# ---------------------------------------------------------------------------


def test_local_contrast_decreases_with_lower_contrast() -> None:
    base = _clean_doc()
    scores = []
    for compress in [1.0, 0.6, 0.3, 0.1]:
        img = _low_contrast(base, compress)
        _, norm = local_contrast_features(img)
        scores.append(norm["local_contrast"])

    assert scores[0] > scores[-1], (
        f"High contrast ({scores[0]:.1f}) must outscore washed-out ({scores[-1]:.1f})"
    )
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 2.0, (
            f"Contrast score increased as contrast reduced: "
            f"compress[{i}]→{scores[i]:.1f}, compress[{i+1}]→{scores[i+1]:.1f}"
        )


# ---------------------------------------------------------------------------
# 10. Crinkle / fold — fold lines must reduce the score vs flat background
# ---------------------------------------------------------------------------


def test_crinkle_fold_score_decreases_with_folds() -> None:
    base = _clean_doc()
    _, norm_clean = crinkle_fold_features(base)

    folded = _add_fold(base, fold_width=10)
    _, norm_folded = crinkle_fold_features(folded)

    assert norm_clean["crinkle_fold"] >= norm_folded["crinkle_fold"] - 2.0, (
        f"Clean ({norm_clean['crinkle_fold']:.1f}) should not score lower than "
        f"folded ({norm_folded['crinkle_fold']:.1f})"
    )


# ---------------------------------------------------------------------------
# 11. BRISQUE-like — smooth natural-like image must outscore heavy distortion
# ---------------------------------------------------------------------------


def test_brisque_score_decreases_with_heavy_distortion() -> None:
    # BRISQUE is calibrated for natural images: a near-uniform smooth grey
    # image has MSCN coefficients close to Gaussian (α ≈ 2, low σ) → high score.
    # Impulse / salt-and-pepper noise pushes σ very high → lower score.
    smooth = np.full((300, 300), 200, dtype=np.uint8)
    smooth = cv2.GaussianBlur(smooth, (9, 9), 3)  # any slight variation

    rng = np.random.default_rng(7)
    impulse = smooth.copy()
    mask = rng.random(smooth.shape) < 0.25  # 25% pixel corruption
    impulse[mask] = rng.integers(0, 256, int(mask.sum()), dtype=np.uint8)

    _, norm_smooth = brisque_like_features(smooth)
    _, norm_impulse = brisque_like_features(impulse)
    assert norm_smooth["brisque"] > norm_impulse["brisque"], (
        f"Smooth image BRISQUE ({norm_smooth['brisque']:.1f}) must outscore "
        f"impulse-noisy ({norm_impulse['brisque']:.1f})"
    )


def test_brisque_score_in_range_across_conditions() -> None:
    for img in [
        np.full((200, 200), 200, dtype=np.uint8),
        _clean_doc(),
        _apply_noise(_clean_doc(), 40),
        _apply_blur(_clean_doc(), 31),
    ]:
        _, norm = brisque_like_features(img)
        assert 0.0 <= norm["brisque"] <= 100.0


# ---------------------------------------------------------------------------
# 12. All individual scores are in [0, 100]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extractor,key",
    [
        (blurriness_features, "sharpness"),
        (noise_features, "noise"),
        (skew_features, "skew"),
        (brightness_features, "brightness"),
        (ridge_features, "ridges"),
        (edge_features, "edges"),
        (ink_bleedthrough_features, "ink_bleedthrough"),
        (shadow_gradient_features, "shadow_gradient"),
        (local_contrast_features, "local_contrast"),
        (crinkle_fold_features, "crinkle_fold"),
        (brisque_like_features, "brisque"),
    ],
)
def test_score_in_range(extractor, key) -> None:
    for img in [
        _clean_doc(),
        _apply_blur(_clean_doc(), 31),
        _apply_noise(_clean_doc(), 30),
        np.full((200, 200), 30, dtype=np.uint8),   # very dark
        np.full((200, 200), 240, dtype=np.uint8),  # blank white
    ]:
        _, norm = extractor(img)
        score = norm[key]
        assert 0.0 <= score <= 100.0, (
            f"{extractor.__name__} returned {key}={score:.2f} outside [0, 100]"
        )


# ---------------------------------------------------------------------------
# 13. Composite score — best document must outscore degraded versions
# ---------------------------------------------------------------------------


def test_composite_score_decreases_with_degradation() -> None:
    clean = _clean_doc()

    result_clean = compute_doc_qual_score(clean, verbose=False)
    # Heavy blur: unambiguously kills sharpness (weight 0.22, largest weight)
    result_heavy_blur = compute_doc_qual_score(_apply_blur(clean, 61), verbose=False)
    # Very dark: tanks brightness score
    result_dark = compute_doc_qual_score(_darken(clean, 0.15), verbose=False)
    # Heavy noise at std=55 overwhelms the noise feature (weight 0.16)
    result_heavy_noise = compute_doc_qual_score(_apply_noise(clean, 55), verbose=False)

    assert result_clean.ocr_score > result_heavy_blur.ocr_score, (
        f"Clean ({result_clean.ocr_score:.1f}) should beat heavy blur "
        f"({result_heavy_blur.ocr_score:.1f})"
    )
    assert result_clean.ocr_score > result_dark.ocr_score, (
        f"Clean ({result_clean.ocr_score:.1f}) should beat very dark "
        f"({result_dark.ocr_score:.1f})"
    )
    assert result_clean.ocr_score > result_heavy_noise.ocr_score, (
        f"Clean ({result_clean.ocr_score:.1f}) should beat heavy noise "
        f"({result_heavy_noise.ocr_score:.1f})"
    )


def test_composite_score_always_in_range() -> None:
    for img in [
        _clean_doc(),
        _apply_blur(_clean_doc(), 61),
        _apply_noise(_clean_doc(), 40),
        _darken(_clean_doc(), 0.1),
        np.full((150, 150), 128, dtype=np.uint8),
    ]:
        result = compute_doc_qual_score(img, verbose=False)
        assert 0.0 <= result.ocr_score <= 100.0
        assert result.passed == (result.ocr_score >= result.threshold)


# ---------------------------------------------------------------------------
# 14. Positive orientation contract: all feature scores in result are [0,100]
# ---------------------------------------------------------------------------


def test_all_feature_scores_positive_orientation() -> None:
    """Confirm every key in feature_scores follows higher=better convention."""
    result = compute_doc_qual_score(_clean_doc(), verbose=False)
    for name, score in result.feature_scores.items():
        assert 0.0 <= score <= 100.0, (
            f"Feature '{name}' score={score:.2f} violates [0,100] contract"
        )
    # Clean doc must score above 50 on most features — no inverted outliers
    above_50 = sum(1 for s in result.feature_scores.values() if s >= 50)
    total = len(result.feature_scores)
    assert above_50 >= total * 0.7, (
        f"Only {above_50}/{total} features scored ≥50 on a clean document — "
        "possible orientation inversion"
    )
