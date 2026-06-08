"""Tests for PDF quality scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from doc_qual.pdf import PageResult, PDFQualityResult, _pypdfium2_available


def _make_minimal_pdf(path: Path) -> None:
    """Write a minimal valid single-page PDF file."""
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
    path.write_bytes(content)


# ── PDFQualityResult dataclass ────────────────────────────────────────────────


def test_page_result_to_dict() -> None:
    pr = PageResult(
        page_number=1,
        ocr_score=72.5,
        passed=True,
        feature_scores={"sharpness": 80.0},
        raw_features={"blur_var": 120.5},
    )
    d = pr.to_dict()
    assert d["page_number"] == 1
    assert d["ocr_score"] == pytest.approx(72.5)
    assert d["passed"] is True


def test_pdf_quality_result_to_dict() -> None:
    import json

    pr = PageResult(page_number=1, ocr_score=65.0, passed=True)
    result = PDFQualityResult(
        path="test.pdf",
        page_count=1,
        pages=[pr],
        summary_score=65.0,
        worst_page=1,
        passed=True,
        threshold=60.0,
    )
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert d["page_count"] == 1
    assert d["summary_score"] == pytest.approx(65.0)


# ── renderer availability ──────────────────────────────────────────────────────


def test_import_error_without_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling score_pdf without any PDF library raises ImportError."""
    monkeypatch.setitem(sys.modules, "pypdfium2", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "pdf2image", None)  # type: ignore[arg-type]

    import importlib

    from doc_qual import pdf as pdf_mod

    importlib.reload(pdf_mod)

    with pytest.raises((ImportError, TypeError)):
        list(pdf_mod._render_pages(Path("dummy.pdf"), dpi=72, max_pages=1))


# ── score_pdf with real renderer ──────────────────────────────────────────────


@pytest.mark.skipif(not _pypdfium2_available(), reason="pypdfium2 not installed")
def test_score_pdf_single_page(tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    _make_minimal_pdf(pdf)
    from doc_qual.pdf import score_pdf

    result = score_pdf(pdf, threshold=60.0)
    assert isinstance(result, PDFQualityResult)
    assert result.page_count >= 1
    assert 0.0 <= result.summary_score <= 100.0
    assert result.worst_page >= 1


@pytest.mark.skipif(not _pypdfium2_available(), reason="pypdfium2 not installed")
def test_score_pdf_missing_file_raises() -> None:
    from doc_qual.pdf import score_pdf

    with pytest.raises(FileNotFoundError):
        score_pdf("/nonexistent/path/doc.pdf")


def test_score_pdf_file_not_found_always() -> None:
    """FileNotFoundError should be raised regardless of renderer availability."""
    from doc_qual.pdf import score_pdf

    with pytest.raises(FileNotFoundError):
        score_pdf("/nonexistent/path/doc.pdf")
