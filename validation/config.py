"""Central configuration for the OCR validation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OCRProviderName = Literal["mock", "tesseract", "easyocr"]
DifficultyLevel = Literal["easy", "medium", "hard", "impossible"]
DocumentType = Literal["invoice", "form", "letter", "receipt", "table", "contract"]
QualityCondition = Literal[
    "clean",
    "blur",
    "noise",
    "skew",
    "low_contrast",
    "faded",
    "watermark",
    "shadow",
    "low_dpi",
]

_DEFAULT_CONDITIONS: list[QualityCondition] = [
    "clean",
    "blur",
    "noise",
    "skew",
    "low_contrast",
    "faded",
    "watermark",
    "shadow",
    "low_dpi",
]

_DEFAULT_TYPES: list[DocumentType] = [
    "invoice",
    "form",
    "letter",
    "receipt",
    "table",
    "contract",
]


@dataclass
class ValidationConfig:
    """All tunable parameters for a validation run."""

    output_dir: Path = field(default_factory=lambda: Path("Validation"))
    n_documents: int = 54
    seed: int = 42
    ocr_provider: OCRProviderName = "mock"
    doc_qual_threshold: float = 60.0
    quality_conditions: list[QualityCondition] = field(
        default_factory=lambda: list(_DEFAULT_CONDITIONS)
    )
    document_types: list[DocumentType] = field(
        default_factory=lambda: list(_DEFAULT_TYPES)
    )
    image_width: int = 900
    image_height: int = 1200
    ocr_timeout_s: int = 30
    confidence_level: float = 0.95
    verbose: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    # ── derived paths ──────────────────────────────────────────────────────
    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "Dataset"

    @property
    def ocr_output_dir(self) -> Path:
        return self.output_dir / "OCR_Output"

    @property
    def quality_index_dir(self) -> Path:
        return self.output_dir / "Quality_Index"

    @property
    def ground_truth_dir(self) -> Path:
        return self.output_dir / "Ground_Truth_Analysis"

    @property
    def hypothesis_dir(self) -> Path:
        return self.output_dir / "Hypothesis_Testing"

    @property
    def research_dir(self) -> Path:
        return self.output_dir / "Research_Artifacts"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "Plots"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "Reports"

    @property
    def paper_dir(self) -> Path:
        return self.output_dir / "Paper_Assets"

    def makedirs(self) -> None:
        for p in (
            self.dataset_dir,
            self.ocr_output_dir,
            self.quality_index_dir,
            self.ground_truth_dir,
            self.hypothesis_dir,
            self.research_dir,
            self.plots_dir,
            self.reports_dir,
            self.paper_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


# ── degradation severity table ─────────────────────────────────────────────
#   Maps condition → (difficulty, mock_wer_rate)
CONDITION_PROFILE: dict[QualityCondition, tuple[DifficultyLevel, float]] = {
    "clean": ("easy", 0.01),
    "blur": ("medium", 0.14),
    "noise": ("medium", 0.18),
    "skew": ("medium", 0.10),
    "low_contrast": ("hard", 0.22),
    "faded": ("hard", 0.28),
    "watermark": ("hard", 0.20),
    "shadow": ("medium", 0.12),
    "low_dpi": ("hard", 0.30),
}
