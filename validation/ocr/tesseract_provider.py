"""Tesseract OCR provider via pytesseract."""

from __future__ import annotations

from pathlib import Path

from validation.ocr.base import OCRProvider, OCRResult


class TesseractProvider(OCRProvider):
    """Wraps pytesseract for English document OCR.

    Requires:
        pip install pytesseract
        brew install tesseract   # macOS
        apt-get install tesseract-ocr  # Debian/Ubuntu
    """

    name = "tesseract"

    def __init__(self, lang: str = "eng", config: str = "--psm 6") -> None:
        self._lang = lang
        self._config = config
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import pytesseract  # noqa: F401

            pytesseract.get_tesseract_version()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _extract(self, image_path: Path, document_id: str) -> OCRResult:
        if not self._available:
            raise RuntimeError(
                "pytesseract or Tesseract binary not found. "
                "Install with: pip install pytesseract && brew install tesseract"
            )

        import cv2
        import pytesseract

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        data = pytesseract.image_to_data(
            img,
            lang=self._lang,
            config=self._config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[str] = []
        word_confs: list[float] = []
        for text, conf in zip(data["text"], data["conf"]):
            text = str(text).strip()
            if text and conf != -1:
                words.append(text)
                word_confs.append(float(conf) / 100.0)

        extracted = " ".join(words)
        avg_conf = sum(word_confs) / max(len(word_confs), 1)

        return OCRResult(
            document_id=document_id,
            provider=self.name,
            extracted_text=extracted,
            confidence=avg_conf,
            word_confidences=word_confs,
            success=True,
        )
