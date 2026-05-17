"""Compute the doc_qual OCR Quality Index for each document image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityIndexResult:
    document_id: str
    ocr_score: float
    passed: bool
    threshold: float
    feature_scores: dict[str, float]
    raw_features: dict[str, float]
    recommendations: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "ocr_quality_index": round(self.ocr_score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "metrics": {
                "blur_score": round(self.feature_scores.get("sharpness", 0.0), 4),
                "noise_score": round(self.feature_scores.get("noise", 0.0), 4),
                "contrast_score": round(self.feature_scores.get("brightness", 0.0), 4),
                "edge_score": round(self.feature_scores.get("edges", 0.0), 4),
                "skew_score": round(self.feature_scores.get("skew", 0.0), 4),
                "ridge_score": round(self.feature_scores.get("ridges", 0.0), 4),
            },
            "raw_features": {k: round(v, 6) for k, v in self.raw_features.items()},
            "recommendations": self.recommendations,
        }


def compute_quality_index(
    image_path: Path,
    document_id: str,
    threshold: float = 60.0,
) -> QualityIndexResult:
    """Run doc_qual on a single image and return a QualityIndexResult."""
    from doc_qual.scorer import compute_doc_qual_score

    result = compute_doc_qual_score(image_path, threshold=threshold, verbose=False)
    return QualityIndexResult(
        document_id=document_id,
        ocr_score=result.ocr_score,
        passed=result.passed,
        threshold=threshold,
        feature_scores=dict(result.feature_scores),
        raw_features=dict(result.raw_features),
        recommendations=list(result.recommendations),
    )
