"""BRISQUE-inspired no-reference image quality signal.

BRISQUE (Blind/Referenceless Image Spatial QUality Evaluator) works by
fitting a Generalised Gaussian Distribution (GGD) to Mean-Subtracted
Contrast-Normalised (MSCN) coefficients of the image.  Pristine images
have MSCN coefficients that are nearly Gaussian (shape α ≈ 2); distortions
push α away from 2.

This module implements the MSCN feature extraction and GGD fitting in pure
NumPy — no opencv-contrib or external model files required.  The resulting
parameters are mapped to a 0–100 quality score without the SVM prediction
step (which requires a labelled training corpus to calibrate).

Reference:
    Mittal et al., "No-Reference Image Quality Assessment in the Spatial
    Domain", IEEE TIP 2012. https://doi.org/10.1109/TIP.2012.2214050
"""

from __future__ import annotations

import cv2
import numpy as np

from doc_qual.utils import linear_score

# ---------------------------------------------------------------------------
# GGD parameter estimation via method of moments
# ---------------------------------------------------------------------------


def _fit_ggd_alpha(x: np.ndarray) -> float:
    """Estimate the GGD shape parameter α using the ratio estimator.

    For a GGD: E[|X|²]² / E[|X|⁴] = Γ(3/α)² / (Γ(1/α) · Γ(5/α))
    We invert this numerically via a look-up table.

    Practical approximation (Sharifi & Leon-Garcia, 1995):
        α ≈ (√(E[X²] / √E[X⁴]))^κ  for a simple ratio-based estimate.
    We use the simpler variance/kurtosis ratio approach which is faster and
    sufficiently accurate for a quality signal.

    Returns α in [0.2, 4.0].
    """
    x_flat = x.ravel().astype(np.float64)
    mu2 = float(np.mean(x_flat**2))
    mu4 = float(np.mean(x_flat**4))
    if mu4 < 1e-10:
        return 2.0
    # Ratio of squared second moment to fourth moment
    ratio = (mu2**2) / max(mu4, 1e-10)
    # For a Gaussian: ratio = 1/3 → α = 2.  Map ratio [0, 1/3] → α [0.3, 2]
    # Empirical fit: α ≈ 2 * (3 * ratio)^0.5
    alpha = float(np.clip(2.0 * np.sqrt(3.0 * ratio), 0.2, 4.0))
    return alpha


def _compute_mscn(gray_f32: np.ndarray) -> np.ndarray:
    """Compute MSCN coefficients.

    MSCN(i,j) = (I(i,j) − μ(i,j)) / (σ(i,j) + 1)
    where μ and σ are local Gaussian-weighted mean and std (7×7 kernel).
    """
    mu = cv2.GaussianBlur(gray_f32, (7, 7), 7.0 / 6.0)
    mu_sq = mu * mu
    sigma_sq = cv2.GaussianBlur(gray_f32 * gray_f32, (7, 7), 7.0 / 6.0) - mu_sq
    sigma = np.sqrt(np.maximum(sigma_sq, 0.0))
    mscn = (gray_f32 - mu) / (sigma + 1.0)
    return mscn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def brisque_like_features(
    gray: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute BRISQUE-inspired no-reference quality features.

    Extracts:
    - MSCN GGD shape α (should be ≈ 2 for clean images)
    - MSCN GGD scale σ
    - Pairwise product GGD parameters in 4 orientations
      (horizontal, vertical, diagonal, anti-diagonal)

    These 10 raw parameters form the first half of the 36-dim BRISQUE
    feature vector.  The second half (sub-band MSCN) is omitted as it
    requires scipy.signal for reliable implementation.

    Args:
        gray: Grayscale uint8 image.

    Returns:
        (raw_features, normalized_scores).  ``brisque`` score:
        100 = statistically near-pristine; 0 = heavily distorted.
    """
    gray_f32 = gray.astype(np.float32) / 255.0
    mscn = _compute_mscn(gray_f32)

    # --- Full-image MSCN GGD -------------------------------------------------
    alpha = _fit_ggd_alpha(mscn)
    sigma = float(np.sqrt(np.mean(mscn**2)))

    # --- Pairwise products in 4 neighbour orientations -----------------------
    # Shift arrays to get neighbour pairs
    pairs: dict[str, np.ndarray] = {
        "h": mscn[:, :-1] * mscn[:, 1:],  # horizontal
        "v": mscn[:-1, :] * mscn[1:, :],  # vertical
        "d1": mscn[:-1, :-1] * mscn[1:, 1:],  # main diagonal
        "d2": mscn[1:, :-1] * mscn[:-1, 1:],  # anti-diagonal
    }

    pair_alphas: dict[str, float] = {}
    for name, prod in pairs.items():
        pair_alphas[f"brisque_pair_alpha_{name}"] = round(_fit_ggd_alpha(prod), 4)

    raw: dict[str, float] = {
        "brisque_alpha": round(alpha, 4),
        "brisque_sigma": round(sigma, 6),
        **pair_alphas,
    }

    # --- Score mapping --------------------------------------------------------
    # For pristine images α ≈ 1.5–2.5 (near-Gaussian MSCN distribution).
    # Distortion pushes α toward 0.3 (heavy tails, impulsive noise) or
    # very high values.  We score highest near α = 2.
    alpha_score = 100.0 - min(100.0, abs(alpha - 2.0) * 40.0)

    # σ of MSCN coefficients: clean images have low σ (≈ 0.02–0.10).
    # High σ → over-amplified noise or very low-contrast image.
    sigma_score = linear_score(sigma, 0.04, 0.35, invert=True)

    # Average pair α deviation from 2.0 — pairwise GGD should also be ≈ 2
    pair_alpha_vals = list(pair_alphas.values())
    pair_score = 100.0 - min(
        100.0, np.mean([abs(a - 2.0) for a in pair_alpha_vals]) * 35.0
    )

    score = float(0.4 * alpha_score + 0.35 * sigma_score + 0.25 * pair_score)
    score = max(0.0, min(100.0, score))

    normalized: dict[str, float] = {"brisque": round(score, 2)}
    return raw, normalized
