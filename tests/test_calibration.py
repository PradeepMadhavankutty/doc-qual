"""Tests for calibration data logging and weight fitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_qual.calibration import (
    append_calibration_row,
    calibration_summary,
    fit_weights_from_calibration,
    load_calibration_data,
)


def _dummy_features(seed: int = 0) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    keys = [
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
    ]
    return {k: rng.uniform(20.0, 95.0) for k in keys}


# ── append & load ─────────────────────────────────────────────────────────────


def test_append_creates_file(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    append_calibration_row("img.png", _dummy_features(), cer=0.05, csv_path=csv)
    assert csv.exists()


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    append_calibration_row(
        "a.png", _dummy_features(0), cer=0.10, engine="tesseract", csv_path=csv
    )
    append_calibration_row(
        "b.png", _dummy_features(1), cer=0.20, engine="tesseract", csv_path=csv
    )
    rows = load_calibration_data(csv_path=csv)
    assert len(rows) == 2
    assert rows[0]["cer"] == pytest.approx(0.10, abs=1e-5)
    assert rows[0]["engine"] == "tesseract"


def test_load_filter_by_engine(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    append_calibration_row(
        "a.png", _dummy_features(0), cer=0.10, engine="tesseract", csv_path=csv
    )
    append_calibration_row(
        "b.png", _dummy_features(1), cer=0.20, engine="azure", csv_path=csv
    )
    rows = load_calibration_data(csv_path=csv, engine="tesseract")
    assert len(rows) == 1
    assert rows[0]["engine"] == "tesseract"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_calibration_data(csv_path=tmp_path / "nonexistent.csv")


# ── fit_weights ───────────────────────────────────────────────────────────────


def _populate_csv(csv: Path, n: int = 25) -> None:
    """Write n synthetic calibration rows with a known pattern."""
    import random

    rng = random.Random(99)
    for i in range(n):
        feats = _dummy_features(i)
        # Make sharpness linearly predict quality (low sharpness → high CER)
        cer = max(
            0.0, min(1.0, 1.0 - feats["sharpness"] / 100.0 + rng.uniform(-0.05, 0.05))
        )
        append_calibration_row(f"img_{i}.png", feats, cer=cer, csv_path=csv)


def test_fit_weights_returns_normalised_dict(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    _populate_csv(csv, 25)
    weights = fit_weights_from_calibration(csv_path=csv, min_rows=20)
    assert isinstance(weights, dict)
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-4)


def test_fit_weights_all_non_negative(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    _populate_csv(csv, 25)
    weights = fit_weights_from_calibration(csv_path=csv, min_rows=20)
    for k, v in weights.items():
        assert v >= 0.0, f"Weight for {k} is negative: {v}"


def test_fit_weights_requires_min_rows(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    _populate_csv(csv, 5)
    with pytest.raises(ValueError, match="at least"):
        fit_weights_from_calibration(csv_path=csv, min_rows=20)


def test_fit_weights_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fit_weights_from_calibration(csv_path=tmp_path / "no.csv")


# ── calibration_summary ───────────────────────────────────────────────────────


def test_summary_no_file(tmp_path: Path) -> None:
    s = calibration_summary(csv_path=tmp_path / "no.csv")
    assert s["exists"] is False
    assert s["total_rows"] == 0


def test_summary_with_data(tmp_path: Path) -> None:
    csv = tmp_path / "cal.csv"
    _populate_csv(csv, 10)
    s = calibration_summary(csv_path=csv)
    assert s["exists"] is True
    assert s["total_rows"] == 10
    assert s["cer_mean"] is not None
