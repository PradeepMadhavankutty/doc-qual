"""Abstract OCR provider interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRResult:
    document_id: str
    provider: str
    extracted_text: str
    confidence: float
    word_confidences: list[float] = field(default_factory=list)
    line_confidences: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.extracted_text.split())

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "provider": self.provider,
            "extracted_text": self.extracted_text,
            "confidence": round(self.confidence, 4),
            "word_count": self.word_count,
            "avg_word_confidence": (
                round(sum(self.word_confidences) / len(self.word_confidences), 4)
                if self.word_confidences
                else None
            ),
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
            "error": self.error,
        }


class OCRProvider(ABC):
    """Base class for all OCR providers."""

    name: str = "base"

    def run(self, image_path: Path, document_id: str) -> OCRResult:
        t0 = time.perf_counter()
        try:
            result = self._extract(image_path, document_id)
        except Exception as exc:  # noqa: BLE001
            result = OCRResult(
                document_id=document_id,
                provider=self.name,
                extracted_text="",
                confidence=0.0,
                success=False,
                error=str(exc),
            )
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    @abstractmethod
    def _extract(self, image_path: Path, document_id: str) -> OCRResult:
        """Subclasses implement OCR extraction here."""
