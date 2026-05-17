"""OCR provider registry — resolve a provider name to a concrete instance."""

from __future__ import annotations

from validation.ocr.base import OCRProvider


def get_provider(name: str, seed: int = 42, **kwargs: object) -> OCRProvider:
    """Return an OCRProvider instance for the given provider name.

    Args:
        name: One of ``"mock"``, ``"tesseract"``, ``"easyocr"``.
        seed: Random seed used by the mock provider.
        **kwargs: Extra keyword arguments forwarded to the provider constructor.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    if name == "mock":
        from validation.ocr.mock_provider import MockOCRProvider

        return MockOCRProvider(seed=seed)

    if name == "tesseract":
        from validation.ocr.tesseract_provider import TesseractProvider

        return TesseractProvider(**kwargs)  # type: ignore[arg-type]

    if name == "easyocr":
        try:
            from validation.ocr.easyocr_provider import (
                EasyOCRProvider,  # type: ignore[import]
            )

            return EasyOCRProvider(**kwargs)  # type: ignore[arg-type]
        except ImportError as exc:
            raise ImportError(
                "EasyOCR is not installed. Run: pip install easyocr"
            ) from exc

    raise ValueError(
        f"Unknown OCR provider {name!r}. Choose from: mock, tesseract, easyocr"
    )
