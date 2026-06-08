"""Public API for Doc-Qual."""

from doc_qual.grid import GridResult, score_image_grid
from doc_qual.pdf import PDFQualityResult, score_pdf
from doc_qual.result import OCRQualityResult
from doc_qual.scorer import compute_doc_qual_score

__version__ = "0.2.0"

__all__ = [
    "GridResult",
    "OCRQualityResult",
    "PDFQualityResult",
    "compute_doc_qual_score",
    "score_image_grid",
    "score_pdf",
]
