"""OCR provider abstractions and implementations."""

from validation.ocr.base import OCRProvider, OCRResult
from validation.ocr.registry import get_provider

__all__ = ["OCRProvider", "OCRResult", "get_provider"]
