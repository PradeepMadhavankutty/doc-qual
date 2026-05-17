"""Structured result objects for Doc-Qual scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OCRQualityResult:
    """OCR readiness score and underlying feature measurements.

    Attributes:
        ocr_score: Composite OCR readiness score on a 0-100 scale.
        passed: Whether the score meets the requested threshold.
        threshold: Threshold used to determine ``passed``.
        feature_scores: Normalized 0-100 score for each feature.
        raw_features: Raw feature measurements before normalization.
        weights: Weights used in the composite score.
        recommendations: Practical suggestions for improving the image.
    """

    ocr_score: float
    passed: bool
    threshold: float
    feature_scores: dict[str, float] = field(default_factory=dict)
    raw_features: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "ocr_score": round(self.ocr_score, 2),
            "passed": self.passed,
            "threshold": self.threshold,
            "feature_scores": {
                key: round(value, 2) for key, value in self.feature_scores.items()
            },
            "raw_features": {
                key: round(value, 4) for key, value in self.raw_features.items()
            },
            "weights": self.weights,
            "recommendations": self.recommendations,
        }
