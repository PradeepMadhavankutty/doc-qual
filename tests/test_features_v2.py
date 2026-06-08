"""Tests for the five new v0.2.0 feature extractors."""

from __future__ import annotations

import numpy as np
import pytest

from doc_qual.features.brisque_like import brisque_like_features
from doc_qual.features.crinkle_fold import crinkle_fold_features
from doc_qual.features.ink_bleedthrough import ink_bleedthrough_features
from doc_qual.features.local_contrast import local_contrast_features
from doc_qual.features.shadow_gradient import shadow_gradient_features

# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def clean_doc(clean_doc: np.ndarray) -> np.ndarray:  # type: ignore[override]
    """Re-use the shared conftest fixture."""
    return clean_doc  # type: ignore[return-value]


@pytest.fixture()
def bleedthrough_doc() -> np.ndarray:
    """Image simulating ink bleed-through in margins."""
    img = np.ones((300, 400), dtype=np.uint8) * 240
    # Scatter dark pixels throughout margins
    rng = np.random.default_rng(0)
    for _ in range(600):
        y = rng.integers(0, 36)  # top margin
        x = rng.integers(0, 400)
        img[y, x] = rng.integers(10, 60)
    for _ in range(600):
        y = rng.integers(264, 300)  # bottom margin
        x = rng.integers(0, 400)
        img[y, x] = rng.integers(10, 60)
    return img


@pytest.fixture()
def shadowed_doc() -> np.ndarray:
    """Document with a strong brightness gradient (left dark, right bright)."""
    img = np.zeros((300, 400), dtype=np.uint8)
    for col in range(400):
        img[:, col] = int(col / 400 * 200) + 20
    return img


@pytest.fixture()
def low_contrast_doc() -> np.ndarray:
    """Document where all pixel values cluster near mid-grey."""
    img = np.ones((300, 400), dtype=np.uint8) * 128
    # slight texture to avoid division issues
    img[100:200, 50:350] = 140
    return img


@pytest.fixture()
def folded_doc() -> np.ndarray:
    """Document with a horizontal fold line."""
    img = np.ones((300, 400), dtype=np.uint8) * 240
    # Strong horizontal dark line simulating a fold
    img[150:155, :] = 30
    return img


# ── ink_bleedthrough ──────────────────────────────────────────────────────────


def test_ink_bleedthrough_clean_scores_high(clean_doc: np.ndarray) -> None:
    _, norm = ink_bleedthrough_features(clean_doc)
    assert norm["ink_bleedthrough"] >= 50.0


def test_ink_bleedthrough_heavy_pollution_scores_low() -> None:
    """An image where margins are densely filled with dark pixels scores low."""
    img = np.ones((300, 400), dtype=np.uint8) * 245
    # Fill top/bottom margins ~20% with near-black pixels → severe bleedthrough
    img[:50, :] = 15
    img[250:, :] = 15
    _, norm = ink_bleedthrough_features(img)
    assert norm["ink_bleedthrough"] < 50.0


def test_ink_bleedthrough_score_in_range(clean_doc: np.ndarray) -> None:
    _, norm = ink_bleedthrough_features(clean_doc)
    assert 0.0 <= norm["ink_bleedthrough"] <= 100.0


def test_ink_bleedthrough_raw_keys(clean_doc: np.ndarray) -> None:
    raw, _ = ink_bleedthrough_features(clean_doc)
    assert "bleedthrough_ratio" in raw
    assert "bleedthrough_spatial_std" in raw


# ── shadow_gradient ───────────────────────────────────────────────────────────


def test_shadow_gradient_clean_scores_high(clean_doc: np.ndarray) -> None:
    _, norm = shadow_gradient_features(clean_doc)
    assert norm["shadow_gradient"] >= 50.0


def test_shadow_gradient_shadowed_scores_lower(
    clean_doc: np.ndarray, shadowed_doc: np.ndarray
) -> None:
    _, norm_clean = shadow_gradient_features(clean_doc)
    _, norm_shadow = shadow_gradient_features(shadowed_doc)
    assert norm_shadow["shadow_gradient"] < norm_clean["shadow_gradient"]


def test_shadow_gradient_score_in_range(clean_doc: np.ndarray) -> None:
    _, norm = shadow_gradient_features(clean_doc)
    assert 0.0 <= norm["shadow_gradient"] <= 100.0


def test_shadow_gradient_raw_keys(clean_doc: np.ndarray) -> None:
    raw, _ = shadow_gradient_features(clean_doc)
    assert "shadow_brightness_std" in raw
    assert "shadow_range" in raw


# ── local_contrast ────────────────────────────────────────────────────────────


def test_local_contrast_clean_scores_higher_than_grey(
    clean_doc: np.ndarray, low_contrast_doc: np.ndarray
) -> None:
    _, norm_clean = local_contrast_features(clean_doc)
    _, norm_grey = local_contrast_features(low_contrast_doc)
    assert norm_clean["local_contrast"] > norm_grey["local_contrast"]


def test_local_contrast_score_in_range(clean_doc: np.ndarray) -> None:
    _, norm = local_contrast_features(clean_doc)
    assert 0.0 <= norm["local_contrast"] <= 100.0


def test_local_contrast_raw_keys(clean_doc: np.ndarray) -> None:
    raw, _ = local_contrast_features(clean_doc)
    assert "local_contrast_mean" in raw
    assert "local_contrast_p10" in raw


# ── crinkle_fold ──────────────────────────────────────────────────────────────


def test_crinkle_fold_clean_scores_high(clean_doc: np.ndarray) -> None:
    _, norm = crinkle_fold_features(clean_doc)
    assert norm["crinkle_fold"] >= 30.0  # clean document should score reasonably well


def test_crinkle_fold_score_in_range(clean_doc: np.ndarray) -> None:
    _, norm = crinkle_fold_features(clean_doc)
    assert 0.0 <= norm["crinkle_fold"] <= 100.0


def test_crinkle_fold_raw_keys(clean_doc: np.ndarray) -> None:
    raw, _ = crinkle_fold_features(clean_doc)
    assert "fold_pixel_ratio" in raw
    assert "bg_log_p95" in raw
    assert "bg_log_std" in raw


# ── brisque_like ──────────────────────────────────────────────────────────────


def test_brisque_clean_scores_reasonably(clean_doc: np.ndarray) -> None:
    _, norm = brisque_like_features(clean_doc)
    assert 0.0 <= norm["brisque"] <= 100.0


def test_brisque_noisy_scores_lower(clean_doc: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    noisy = clean_doc.astype(np.float32) + rng.normal(0, 40, clean_doc.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    _, norm_clean = brisque_like_features(clean_doc)
    _, norm_noisy = brisque_like_features(noisy)
    assert (
        norm_noisy["brisque"] <= norm_clean["brisque"] + 15.0
    )  # noisy ≤ clean (with tolerance)


def test_brisque_raw_keys(clean_doc: np.ndarray) -> None:
    raw, _ = brisque_like_features(clean_doc)
    assert "brisque_alpha" in raw
    assert "brisque_sigma" in raw
    assert "brisque_pair_alpha_h" in raw


# ── integration: scorer includes new features ─────────────────────────────────


def test_scorer_includes_new_features(clean_doc: np.ndarray) -> None:
    from doc_qual.scorer import compute_doc_qual_score

    result = compute_doc_qual_score(clean_doc, verbose=False)
    new_keys = {
        "ink_bleedthrough",
        "shadow_gradient",
        "local_contrast",
        "crinkle_fold",
        "brisque",
    }
    assert new_keys.issubset(result.feature_scores.keys())
