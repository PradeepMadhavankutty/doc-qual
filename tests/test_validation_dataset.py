"""Tests for dataset generation and manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.config import ValidationConfig
from validation.dataset.generator import DocumentGenerator
from validation.dataset.manifest import DatasetManifest, DocumentRecord


@pytest.fixture
def small_cfg(tmp_path: Path) -> ValidationConfig:
    return ValidationConfig(
        output_dir=tmp_path / "Validation",
        n_documents=9,
        seed=7,
        document_types=["invoice", "form", "letter"],
        quality_conditions=["clean", "blur", "noise"],
    )


def test_generator_creates_images(small_cfg: ValidationConfig) -> None:
    gen = DocumentGenerator(small_cfg)
    manifest = gen.generate(small_cfg.dataset_dir)
    assert len(manifest) == 9
    for doc in manifest:
        assert Path(doc.image_path).exists()


def test_manifest_summary_counts(small_cfg: ValidationConfig) -> None:
    gen = DocumentGenerator(small_cfg)
    manifest = gen.generate(small_cfg.dataset_dir)
    summary = manifest.summary()
    assert summary["total_documents"] == 9
    assert len(summary["document_types"]) >= 1
    assert len(summary["quality_conditions"]) >= 1


def test_manifest_round_trip(small_cfg: ValidationConfig, tmp_path: Path) -> None:
    gen = DocumentGenerator(small_cfg)
    manifest = gen.generate(small_cfg.dataset_dir)
    save_path = tmp_path / "manifest.json"
    manifest.save(save_path)
    loaded = DatasetManifest.load(save_path)
    assert len(loaded) == len(manifest)
    ids_orig = [r.document_id for r in manifest]
    ids_loaded = [r.document_id for r in loaded]
    assert ids_orig == ids_loaded


def test_ground_truth_is_non_empty(small_cfg: ValidationConfig) -> None:
    gen = DocumentGenerator(small_cfg)
    manifest = gen.generate(small_cfg.dataset_dir)
    for doc in manifest:
        assert len(doc.ground_truth) > 0
        assert len(doc.ground_truth_words) > 0


def test_document_record_to_dict() -> None:
    rec = DocumentRecord(
        document_id="doc_0001",
        image_path="/tmp/img.png",
        document_type="invoice",
        quality_condition="clean",
        difficulty_level="easy",
        source_dataset="test",
        ground_truth="Hello world",
        ground_truth_words=["Hello", "world"],
    )
    d = rec.to_dict()
    assert d["word_count"] == 2
    assert d["char_count"] == len("Hello world")
