"""Synthetic document generator.

Creates realistic document images (invoice, form, letter, receipt, table,
contract) with known ground-truth text, then applies calibrated degradation
conditions so downstream OCR accuracy correlates with doc_qual scores.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from validation.config import CONDITION_PROFILE, ValidationConfig
from validation.dataset.manifest import DatasetManifest, DocumentRecord

if TYPE_CHECKING:
    pass

# ── ground-truth text templates ───────────────────────────────────────────

_TEMPLATES: dict[str, list[str]] = {
    "invoice": [
        "INVOICE",
        "Invoice Number: INV-{idx:04d}",
        "Date: 2026-05-{day:02d}",
        "Bill To: Acme Corporation",
        "123 Business Avenue, Suite 400",
        "Description: Document Processing Service",
        "Quantity: {qty}  Unit Price: ${price:.2f}",
        "Subtotal: ${subtotal:.2f}",
        "Tax (8%): ${tax:.2f}",
        "TOTAL DUE: ${total:.2f}",
        "Payment due within 30 days.",
        "Thank you for your business.",
    ],
    "form": [
        "APPLICATION FORM",
        "Full Name: John A. Smith",
        "Date of Birth: 1985-03-{day:02d}",
        "Address: 456 Main Street, Springfield",
        "Email: john.smith@example.com",
        "Phone: +1 (555) {phone}",
        "Occupation: Software Engineer",
        "Annual Income: ${income:,}",
        "Signature: ________________",
        "Date Signed: 2026-05-{day:02d}",
    ],
    "letter": [
        "RE: Project Update Memorandum",
        "Dear Mr. Johnson,",
        "I am writing to confirm the details of our arrangement.",
        "As discussed, the deliverables for Phase {idx} are on schedule.",
        "The total investment for this engagement is ${total:.2f}.",
        "Please review the attached documentation at your earliest convenience.",
        "We remain committed to delivering the highest quality outcomes.",
        "Should you have any questions, do not hesitate to contact us.",
        "Sincerely,",
        "Dr. Patricia Moore",
        "Director of Operations",
    ],
    "receipt": [
        "SALES RECEIPT",
        "Store: QuickMart #{idx:03d}",
        "Date: 2026-05-{day:02d}  Time: {hour:02d}:{minute:02d}",
        "Item: Premium Coffee    $4.50",
        "Item: Organic Sandwich  $8.75",
        "Item: Sparkling Water   $2.25",
        "Item: Dark Chocolate    $3.99",
        "Subtotal: $19.49",
        "Discount: -$1.95",
        "Sales Tax: $1.40",
        "TOTAL: $18.94",
        "CASH TENDERED: $20.00",
        "CHANGE: $1.06",
        "Thank you! Receipt #{receipt:06d}",
    ],
    "table": [
        "QUARTERLY PERFORMANCE REPORT",
        "Period: Q{quarter} 2026",
        "Department | Revenue | Expenses | Profit",
        "-----------|---------|----------|-------",
        "Sales      | $142500 |  $89200  | $53300",
        "Marketing  |  $67800 |  $71400  | -$3600",
        "Operations |  $98200 |  $64100  | $34100",
        "Support    |  $31400 |  $28900  |  $2500",
        "TOTAL      | $339900 | $253600  | $86300",
        "Year-over-year growth: +{growth}%",
        "Forecast accuracy: {accuracy}%",
    ],
    "contract": [
        "SERVICE AGREEMENT",
        "Agreement No. AGR-{idx:04d}",
        "This Agreement is entered into on 2026-05-{day:02d}",
        "between TechCorp Solutions Inc. (Provider)",
        "and Meridian Industries Ltd. (Client).",
        "1. SCOPE OF SERVICES",
        "Provider agrees to deliver software development services",
        "as outlined in Schedule A attached hereto.",
        "2. PAYMENT TERMS",
        "Client shall pay ${total:.2f} within 30 days of invoice.",
        "3. TERM AND TERMINATION",
        "This Agreement commences on the date first written above.",
        "Either party may terminate with 30 days written notice.",
        "4. CONFIDENTIALITY",
        "Both parties agree to maintain strict confidentiality.",
        "Signed: _________________  Date: 2026-05-{day:02d}",
    ],
}


def _render_text_lines(
    lines: list[str],
    width: int,
    height: int,
    font_scale: float = 0.55,
    thickness: int = 1,
) -> np.ndarray:
    """Render text lines onto a white canvas using cv2.putText."""
    canvas = np.ones((height, width), dtype=np.uint8) * 248
    x_margin = int(width * 0.06)
    y_start = int(height * 0.06)
    line_height = int(height / max(len(lines) + 4, 20))
    line_height = max(line_height, 28)

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        if y > height - 20:
            break
        cv2.putText(
            canvas,
            line,
            (x_margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            15,
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def _fill_template(template: list[str], rng: random.Random) -> list[str]:
    """Substitute placeholder values in a template."""
    ctx = {
        "idx": rng.randint(1, 9999),
        "day": rng.randint(1, 28),
        "hour": rng.randint(8, 20),
        "minute": rng.randint(0, 59),
        "qty": rng.randint(1, 50),
        "price": rng.uniform(10.0, 500.0),
        "subtotal": rng.uniform(100.0, 5000.0),
        "tax": rng.uniform(8.0, 400.0),
        "total": rng.uniform(108.0, 5400.0),
        "phone": f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        "income": rng.randint(40000, 200000),
        "receipt": rng.randint(1, 999999),
        "quarter": rng.randint(1, 4),
        "growth": rng.randint(2, 25),
        "accuracy": rng.randint(85, 99),
    }
    result = []
    for line in template:
        try:
            result.append(line.format(**ctx))
        except (KeyError, ValueError):
            result.append(line)
    return result


# ── degradation functions ─────────────────────────────────────────────────


def _degrade(
    image: np.ndarray,
    condition: str,
    strength: float,
    rng_np: np.random.Generator,
) -> np.ndarray:
    """Apply a single degradation condition at the given strength [0..1]."""
    img = image.astype(np.float32)

    if condition == "clean":
        pass

    elif condition == "blur":
        k = int(3 + strength * 18)
        k = k if k % 2 == 1 else k + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif condition == "noise":
        sigma = 5.0 + strength * 45.0
        img = img + rng_np.normal(0, sigma, img.shape)

    elif condition == "skew":
        angle = strength * 12.0
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=248.0,
        )

    elif condition == "low_contrast":
        alpha = 1.0 - strength * 0.55
        img = img * alpha + 128.0 * (1.0 - alpha)

    elif condition == "faded":
        img = img * (1.0 - strength * 0.45) + 230.0 * (strength * 0.45)

    elif condition == "watermark":
        h, w = img.shape
        overlay = np.ones_like(img) * 200.0
        text_y = h // 2
        cv2.putText(
            overlay,
            "CONFIDENTIAL",
            (int(w * 0.1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            100,
            4,
            cv2.LINE_AA,
        )
        alpha = 0.15 + strength * 0.25
        img = img * (1.0 - alpha) + overlay * alpha

    elif condition == "shadow":
        h, w = img.shape
        grad = np.linspace(0, 1, w, dtype=np.float32)
        shadow_map = (1.0 - grad * strength * 0.5)[np.newaxis, :]
        img = img * shadow_map

    elif condition == "low_dpi":
        h, w = img.shape
        scale = max(0.2, 1.0 - strength * 0.65)
        small = cv2.resize(
            img.astype(np.uint8),
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        img = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


# ── public generator ──────────────────────────────────────────────────────

_DIFFICULTY_STRENGTHS: dict[str, float] = {
    "clean": 0.0,
    "blur": 0.55,
    "noise": 0.50,
    "skew": 0.45,
    "low_contrast": 0.55,
    "faded": 0.60,
    "watermark": 0.65,
    "shadow": 0.55,
    "low_dpi": 0.65,
}


class DocumentGenerator:
    """Generate a balanced synthetic dataset and write images + manifest."""

    def __init__(self, cfg: ValidationConfig) -> None:
        self._cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._np_rng = np.random.default_rng(cfg.seed)

    def generate(self, output_dir: Path) -> DatasetManifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = DatasetManifest()

        doc_types = self._cfg.document_types
        conditions = self._cfg.quality_conditions
        total = self._cfg.n_documents

        # Build a balanced spec list cycling through type×condition pairs
        combos = [(dt, cond) for dt in doc_types for cond in conditions]
        specs = []
        while len(specs) < total:
            specs.extend(combos)
        self._rng.shuffle(specs)
        specs = specs[:total]

        for idx, (doc_type, condition) in enumerate(specs):
            doc_id = f"doc_{idx + 1:04d}"
            difficulty, _ = CONDITION_PROFILE[condition]  # type: ignore[index]
            lines = _fill_template(_TEMPLATES[doc_type], self._rng)
            ground_truth = " ".join(lines)

            canvas = _render_text_lines(
                lines,
                self._cfg.image_width,
                self._cfg.image_height,
            )
            strength = _DIFFICULTY_STRENGTHS.get(condition, 0.5)
            degraded = _degrade(canvas, condition, strength, self._np_rng)

            image_path = output_dir / f"{doc_id}_{doc_type}_{condition}.png"
            cv2.imwrite(str(image_path), degraded)

            manifest.add(
                DocumentRecord(
                    document_id=doc_id,
                    image_path=str(image_path),
                    document_type=doc_type,
                    quality_condition=condition,
                    difficulty_level=difficulty,
                    source_dataset="synthetic_doc_qual_v1",
                    ground_truth=ground_truth,
                    ground_truth_words=ground_truth.split(),
                )
            )

        return manifest

    @staticmethod
    def compute_expected_quality(condition: str) -> float:
        """Return a 0-100 proxy for expected quality given a condition."""
        _quality_map = {
            "clean": 92.0,
            "shadow": 78.0,
            "skew": 72.0,
            "blur": 60.0,
            "noise": 58.0,
            "watermark": 55.0,
            "low_contrast": 48.0,
            "faded": 42.0,
            "low_dpi": 35.0,
        }
        return _quality_map.get(condition, 60.0)

    @staticmethod
    def estimate_expected_wer(condition: str) -> float:
        """Return the expected WER for a condition (used to validate mock OCR)."""
        _, wer = CONDITION_PROFILE.get(condition, ("medium", 0.20))  # type: ignore[call-overload]
        return wer
