"""PDF document quality scoring.

Renders each page of a PDF to a grayscale image and scores it with the
full doc-qual feature pipeline.  Returns per-page results and a summary
score (median across pages, robust to one bad page).

Requires ``pypdfium2`` (recommended, pure-Python wheel) or
``pdf2image`` + poppler (system dependency).

Usage::

    from doc_qual.pdf import score_pdf

    result = score_pdf("contract.pdf")
    print(result.summary_score)   # median quality across all pages
    print(result.worst_page)      # 1-indexed page number with lowest score
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── result dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PageResult:
    """Quality score for a single PDF page."""

    page_number: int  # 1-indexed
    ocr_score: float
    passed: bool
    feature_scores: dict[str, float] = field(default_factory=dict)
    raw_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "ocr_score": round(self.ocr_score, 2),
            "passed": self.passed,
            "feature_scores": {k: round(v, 2) for k, v in self.feature_scores.items()},
            "raw_features": {k: round(v, 4) for k, v in self.raw_features.items()},
        }


@dataclass(frozen=True)
class PDFQualityResult:
    """Quality assessment for an entire PDF document."""

    path: str
    page_count: int
    pages: list[PageResult]
    summary_score: float  # median page score (robust to single bad page)
    worst_page: int  # 1-indexed
    passed: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "page_count": self.page_count,
            "summary_score": round(self.summary_score, 2),
            "worst_page": self.worst_page,
            "passed": self.passed,
            "threshold": self.threshold,
            "pages": [p.to_dict() for p in self.pages],
        }


# ── renderer helpers ─────────────────────────────────────────────────────────


def _pypdfium2_available() -> bool:
    try:
        import pypdfium2  # noqa: F401

        return True
    except ImportError:
        return False


def _pdf2image_available() -> bool:
    try:
        import pdf2image  # noqa: F401

        return True
    except ImportError:
        return False


def _render_pages_pypdfium2(
    pdf_path: Path,
    dpi: int,
    max_pages: int | None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    import pypdfium2  # type: ignore[import-untyped]

    doc = pypdfium2.PdfDocument(str(pdf_path))
    scale = dpi / 72.0  # pypdfium2 renders at 72 DPI by default
    n = len(doc) if max_pages is None else min(len(doc), max_pages)
    for i in range(n):
        page = doc[i]
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        gray = np.array(pil_img.convert("L"))
        yield i + 1, gray
    doc.close()


def _render_pages_pdf2image(
    pdf_path: Path,
    dpi: int,
    max_pages: int | None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    from pdf2image import convert_from_path  # type: ignore[import-untyped]

    kwargs: dict[str, Any] = {"dpi": dpi, "grayscale": True}
    if max_pages is not None:
        kwargs["last_page"] = max_pages
    images = convert_from_path(str(pdf_path), **kwargs)
    for i, img in enumerate(images):
        yield i + 1, np.array(img)


def _render_pages(
    pdf_path: Path,
    dpi: int,
    max_pages: int | None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield (1-indexed page_number, grayscale_array) pairs."""
    if _pypdfium2_available():
        yield from _render_pages_pypdfium2(pdf_path, dpi, max_pages)
    elif _pdf2image_available():
        yield from _render_pages_pdf2image(pdf_path, dpi, max_pages)
    else:
        raise ImportError(
            "PDF support requires 'pypdfium2' (recommended) or 'pdf2image'.\n"
            "Install with:\n"
            "  pip install doc-qual[pdf]           # pypdfium2\n"
            "  pip install doc-qual[pdf-poppler]   # pdf2image + poppler"
        )


# ── public API ────────────────────────────────────────────────────────────────


def score_pdf(
    pdf_path: str | Path,
    *,
    threshold: float = 60.0,
    weights: dict[str, float] | None = None,
    engine: str | None = None,
    dpi: int = 150,
    max_pages: int | None = None,
    verbose: bool = False,
) -> PDFQualityResult:
    """Score every page of a PDF and return a :class:`PDFQualityResult`.

    Args:
        pdf_path: Path to the PDF file.
        threshold: Passing score threshold (applied per page and summary).
        weights: Optional custom feature weight overrides.
        engine: Engine profile name (e.g. ``'tesseract'``).
        dpi: Render resolution.  150 is sufficient for quality assessment;
            use 300 for OCR-grade output.
        max_pages: If set, only score the first N pages.
        verbose: Print per-page scores during processing.

    Returns:
        :class:`PDFQualityResult` with per-page and summary scores.
    """
    # Import here to avoid circular dependency
    from doc_qual.scorer import compute_doc_qual_score

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageResult] = []
    for page_num, gray in _render_pages(pdf_path, dpi, max_pages):
        try:
            r = compute_doc_qual_score(
                gray,
                threshold=threshold,
                weights=weights,
                engine=engine,
                verbose=False,
            )
            page_result = PageResult(
                page_number=page_num,
                ocr_score=r.ocr_score,
                passed=r.passed,
                feature_scores=dict(r.feature_scores),
                raw_features=dict(r.raw_features),
            )
        except Exception as exc:  # noqa: BLE001
            page_result = PageResult(
                page_number=page_num,
                ocr_score=0.0,
                passed=False,
                feature_scores={},
                raw_features={"error": str(exc)},  # type: ignore[dict-item]
            )
        pages.append(page_result)
        if verbose:
            status = "✓" if page_result.passed else "✗"
            print(f"  Page {page_num:3d}: {page_result.ocr_score:5.1f}  {status}")

    if not pages:
        return PDFQualityResult(
            path=str(pdf_path),
            page_count=0,
            pages=[],
            summary_score=0.0,
            worst_page=0,
            passed=False,
            threshold=threshold,
        )

    scores = [p.ocr_score for p in pages]
    summary_score = float(np.median(scores))
    worst_page = pages[int(np.argmin(scores))].page_number

    return PDFQualityResult(
        path=str(pdf_path),
        page_count=len(pages),
        pages=pages,
        summary_score=summary_score,
        worst_page=worst_page,
        passed=summary_score >= threshold,
        threshold=threshold,
    )
