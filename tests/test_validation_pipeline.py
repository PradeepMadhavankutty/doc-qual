"""Integration test — runs the full pipeline on a tiny dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.config import ValidationConfig
from validation.pipeline import ValidationPipeline


@pytest.fixture
def mini_cfg(tmp_path: Path) -> ValidationConfig:
    return ValidationConfig(
        output_dir=tmp_path / "Validation",
        n_documents=6,
        seed=99,
        ocr_provider="mock",
        document_types=["invoice", "letter"],
        quality_conditions=["clean", "blur", "noise"],
        verbose=False,
    )


def test_pipeline_runs_end_to_end(mini_cfg: ValidationConfig) -> None:
    report = ValidationPipeline(mini_cfg).run()
    assert len(report.ocr_results) == 6
    assert len(report.quality_results) == 6
    assert len(report.accuracy_results) == 6
    assert report.hypothesis_cer is not None
    assert report.hypothesis_wer is not None
    assert report.elapsed_s > 0


def test_pipeline_artifacts_created(mini_cfg: ValidationConfig) -> None:
    report = ValidationPipeline(mini_cfg).run()
    for p in report.artifact_paths:
        if isinstance(p, Path):
            assert p.exists(), f"Missing artifact: {p}"


def test_pipeline_quality_stats_in_range(mini_cfg: ValidationConfig) -> None:
    report = ValidationPipeline(mini_cfg).run()
    q = report.quality_stats()
    assert 0 <= q["mean"] <= 100
    assert 0 <= q["pass_rate"] <= 1


def test_pipeline_accuracy_stats_in_range(mini_cfg: ValidationConfig) -> None:
    report = ValidationPipeline(mini_cfg).run()
    a = report.accuracy_stats()
    assert 0 <= a["cer_mean"] <= 1
    assert 0 <= a["wer_mean"] <= 1


def test_mock_ocr_all_succeed(mini_cfg: ValidationConfig) -> None:
    report = ValidationPipeline(mini_cfg).run()
    assert all(r.success for r in report.ocr_results)
