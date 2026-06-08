"""Tests for engine-specific weight profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_qual.profiles import ENGINE_PROFILES, load_custom_profile, load_engine_profile


def test_all_built_in_profiles_load() -> None:
    for engine in ENGINE_PROFILES:
        weights = load_engine_profile(engine)
        assert isinstance(weights, dict)
        assert len(weights) > 0


@pytest.mark.parametrize("engine", ENGINE_PROFILES)
def test_profile_weights_are_non_negative(engine: str) -> None:
    weights = load_engine_profile(engine)
    for k, v in weights.items():
        assert v >= 0.0, f"{engine}: {k}={v} is negative"


@pytest.mark.parametrize("engine", ENGINE_PROFILES)
def test_profile_contains_required_keys(engine: str) -> None:
    required = {"sharpness", "noise", "edges", "skew", "brightness", "ridges"}
    weights = load_engine_profile(engine)
    assert required.issubset(weights.keys())


def test_unknown_engine_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        load_engine_profile("nonexistent_engine_xyz")


def test_custom_profile_from_file(tmp_path: Path) -> None:
    profile = {
        "sharpness": 0.30,
        "noise": 0.20,
        "edges": 0.15,
        "skew": 0.15,
        "brightness": 0.10,
        "ridges": 0.10,
    }
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(profile))
    loaded = load_custom_profile(p)
    assert loaded["sharpness"] == pytest.approx(0.30)


def test_custom_profile_missing_required_key_raises(tmp_path: Path) -> None:
    incomplete = {"sharpness": 0.5, "noise": 0.5}  # missing required keys
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="missing required feature keys"):
        load_custom_profile(p)


def test_custom_profile_negative_weight_raises(tmp_path: Path) -> None:
    profile = {
        "sharpness": -0.1,
        "noise": 0.20,
        "edges": 0.15,
        "skew": 0.15,
        "brightness": 0.10,
        "ridges": 0.10,
    }
    p = tmp_path / "neg.json"
    p.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="negative weights"):
        load_custom_profile(p)


def test_scorer_accepts_engine_arg(clean_doc) -> None:  # type: ignore[no-untyped-def]
    from doc_qual.scorer import compute_doc_qual_score

    result = compute_doc_qual_score(clean_doc, engine="tesseract", verbose=False)
    assert result.engine == "tesseract"
    assert result.ocr_score >= 0.0


@pytest.mark.parametrize("engine", ENGINE_PROFILES)
def test_all_engines_score_clean_doc(clean_doc, engine: str) -> None:  # type: ignore[no-untyped-def]
    from doc_qual.scorer import compute_doc_qual_score

    result = compute_doc_qual_score(clean_doc, engine=engine, verbose=False)
    assert 0.0 <= result.ocr_score <= 100.0
    assert result.engine == engine
