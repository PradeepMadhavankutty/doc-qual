"""Public API for Doc-Qual."""

from doc_qual.result import OCRQualityResult
from doc_qual.scorer import compute_doc_qual_score

__version__ = "0.1.0"

__all__ = ["OCRQualityResult", "compute_doc_qual_score"]
