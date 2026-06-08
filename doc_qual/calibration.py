"""Calibration data logging and empirical weight fitting.

Usage — logging::

    from doc_qual.calibration import append_calibration_row
    append_calibration_row("scan.jpg", result.feature_scores, cer=0.12, engine="tesseract")

Usage — fitting weights from accumulated data::

    from doc_qual.calibration import fit_weights_from_calibration
    weights = fit_weights_from_calibration(engine="tesseract")

The calibration CSV lives at ``~/.doc_qual/calibration.csv`` by default.
Each row is one scored image paired with its ground-truth CER.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

DEFAULT_CALIBRATION_PATH: Path = Path.home() / ".doc_qual" / "calibration.csv"

# All possible feature columns (superset — missing features default to NaN).
_FEATURE_COLS: tuple[str, ...] = (
    "sharpness",
    "noise",
    "edges",
    "skew",
    "brightness",
    "ridges",
    "ink_bleedthrough",
    "shadow_gradient",
    "local_contrast",
    "crinkle_fold",
    "brisque",
)

CALIBRATION_COLUMNS: tuple[str, ...] = (
    "image_path",
    "engine",
    *_FEATURE_COLS,
    "cer",
    "timestamp",
)


# ── write ───────────────────────────────────────────────────────────────────


def append_calibration_row(
    image_path: str | Path,
    feature_scores: dict[str, float],
    cer: float,
    engine: str = "default",
    csv_path: Path = DEFAULT_CALIBRATION_PATH,
) -> None:
    """Append a (feature_vector, CER) pair to the calibration CSV.

    Creates ``~/.doc_qual/`` and the CSV header on first use.

    Args:
        image_path: Path to the scored image (stored for provenance only).
        feature_scores: Normalised 0–100 feature scores from
            ``OCRQualityResult.feature_scores``.
        cer: Character Error Rate measured against ground truth (0.0–1.0).
        engine: OCR engine name (e.g. ``'tesseract'``).
        csv_path: Override the default calibration file location.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    row: dict[str, Any] = {
        "image_path": str(image_path),
        "engine": engine,
        "cer": round(cer, 6),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for col in _FEATURE_COLS:
        row[col] = round(feature_scores.get(col, float("nan")), 4)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CALIBRATION_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── read ────────────────────────────────────────────────────────────────────


def load_calibration_data(
    csv_path: Path = DEFAULT_CALIBRATION_PATH,
    engine: str | None = None,
) -> list[dict[str, Any]]:
    """Load calibration rows, optionally filtered by engine.

    Args:
        csv_path: Path to the calibration CSV.
        engine: If given, only rows with this engine value are returned.

    Returns:
        List of row dicts with numeric feature values converted to float.

    Raises:
        FileNotFoundError: if the CSV does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {csv_path}. "
            "Run 'doc-qual calibrate' on some images first."
        )

    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if engine is not None and row.get("engine") != engine:
                continue
            parsed: dict[str, Any] = {
                "image_path": row.get("image_path", ""),
                "engine": row.get("engine", "default"),
                "timestamp": row.get("timestamp", ""),
                "cer": float(row.get("cer", "nan")),
            }
            for col in _FEATURE_COLS:
                raw = row.get(col, "")
                try:
                    parsed[col] = float(raw) if raw not in ("", "nan") else float("nan")
                except ValueError:
                    parsed[col] = float("nan")
            rows.append(parsed)
    return rows


# ── fit ─────────────────────────────────────────────────────────────────────


def fit_weights_from_calibration(
    csv_path: Path = DEFAULT_CALIBRATION_PATH,
    engine: str | None = None,
    feature_names: tuple[str, ...] | None = None,
    min_rows: int = 20,
) -> dict[str, float]:
    """Fit feature weights from accumulated calibration data.

    Uses Non-Negative Least Squares (NNLS) regression to find weights
    ``w`` such that ``feature_matrix @ w ≈ (1 − CER)`` (higher feature
    scores should predict lower error → higher OCR quality).

    If ``scipy`` is unavailable the function falls back to
    ``numpy.linalg.lstsq`` followed by clipping negatives to zero and
    renormalising — a reasonable approximation for typical OCR data.

    Args:
        csv_path: Path to the calibration CSV.
        engine: Restrict fitting to rows for this engine.
        feature_names: Override which features to include.  Defaults to
            all columns present in the CSV that appear in ``_FEATURE_COLS``.
        min_rows: Minimum number of valid rows required.

    Returns:
        Weight dict normalised to sum to 1.0.

    Raises:
        ValueError: if fewer than ``min_rows`` valid rows are available.
        FileNotFoundError: if the calibration file does not exist.
    """
    import numpy as np

    rows = load_calibration_data(csv_path, engine=engine)
    if len(rows) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} calibration rows, got {len(rows)}. "
            "Run 'doc-qual calibrate' on more images."
        )

    cols = feature_names if feature_names else _FEATURE_COLS

    # Build X (feature matrix) and y (OCR quality = 1 − CER)
    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    for row in rows:
        feat_vec = [row.get(c, float("nan")) for c in cols]
        if any(v != v for v in feat_vec):  # NaN check
            continue
        cer = row.get("cer", float("nan"))
        if cer != cer or not (0.0 <= cer <= 1.0):
            continue
        X_rows.append(feat_vec)
        y_vals.append(1.0 - cer)

    if len(X_rows) < min_rows:
        raise ValueError(
            f"After dropping rows with missing values, only {len(X_rows)} rows remain "
            f"(need {min_rows}). Run 'doc-qual calibrate' on more images."
        )

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.float64)

    # Normalise features to [0, 1] range (they are already 0–100; divide by 100)
    X = X / 100.0

    # Fit with NNLS (preferred) or lstsq fallback
    try:
        from scipy.optimize import nnls  # type: ignore[import-untyped]

        weights_raw, _ = nnls(X, y)
    except ImportError:
        # Fallback: unconstrained lstsq then clip negatives
        weights_raw, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        weights_raw = np.clip(weights_raw, 0.0, None)

    total = float(weights_raw.sum())
    if total <= 0:
        # All-zero solution: fall back to equal weights
        weights_raw = np.ones(len(cols), dtype=np.float64)
        total = float(weights_raw.sum())

    normalised = weights_raw / total
    return {col: round(float(w), 6) for col, w in zip(cols, normalised)}


# ── summary ─────────────────────────────────────────────────────────────────


def calibration_summary(
    csv_path: Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, Any]:
    """Return a brief summary of the calibration dataset."""
    if not csv_path.exists():
        return {"exists": False, "path": str(csv_path), "total_rows": 0}

    rows = load_calibration_data(csv_path)
    engines: dict[str, int] = {}
    for row in rows:
        eng = str(row.get("engine", "default"))
        engines[eng] = engines.get(eng, 0) + 1

    cers = [row["cer"] for row in rows if row["cer"] == row["cer"]]
    return {
        "exists": True,
        "path": str(csv_path),
        "total_rows": len(rows),
        "engines": engines,
        "cer_mean": round(sum(cers) / len(cers), 4) if cers else None,
        "cer_min": round(min(cers), 4) if cers else None,
        "cer_max": round(max(cers), 4) if cers else None,
    }
