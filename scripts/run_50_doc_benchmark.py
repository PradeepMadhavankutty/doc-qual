"""Run a 50-document Doc-Qual benchmark and write CSV outputs.

The default benchmark generates synthetic document images with known degradation
levels. This gives the project a reproducible validation sample without needing
large external OCR models or datasets.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from doc_qual import compute_doc_qual_score

GROUND_TRUTH_LINES = [
    "Invoice number INV-2048",
    "Document quality assessment sample",
    "Subtotal amount 1432.50",
    "Payment due within thirty days",
    "Reference code DOCQUAL BENCHMARK",
]


@dataclass(frozen=True)
class DocumentSpec:
    document_id: str
    degradation_level: int
    blur_kernel: int
    noise_sigma: float
    skew_degrees: float
    brightness_shift: int


def create_base_document(width: int = 900, height: int = 1200) -> np.ndarray:
    image = np.ones((height, width), dtype=np.uint8) * 242
    y = 140
    for index, line in enumerate(GROUND_TRUTH_LINES):
        cv2.putText(
            image,
            line,
            (90, y + index * 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            25,
            3,
            cv2.LINE_AA,
        )
    for y_line in range(730, 980, 55):
        cv2.line(image, (90, y_line), (810, y_line), 35, 3)
    cv2.rectangle(image, (70, 90), (830, 1030), 80, 3)
    return image


def build_specs(count: int, seed: int) -> list[DocumentSpec]:
    rng = random.Random(seed)
    specs: list[DocumentSpec] = []
    for index in range(count):
        level = index % 5
        specs.append(
            DocumentSpec(
                document_id=f"doc_{index + 1:03d}",
                degradation_level=level,
                blur_kernel=[1, 3, 7, 11, 17][level],
                noise_sigma=[0.0, 5.0, 11.0, 18.0, 28.0][level],
                skew_degrees=rng.uniform(-1, 1) + [0.0, 1.5, -3.0, 5.0, -8.0][level],
                brightness_shift=[0, -8, 15, -28, 38][level],
            )
        )
    rng.shuffle(specs)
    return specs


def degrade_document(base: np.ndarray, spec: DocumentSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = base.astype(np.float32) + spec.brightness_shift
    if spec.blur_kernel > 1:
        image = cv2.GaussianBlur(image, (spec.blur_kernel, spec.blur_kernel), 0)
    if spec.noise_sigma > 0:
        image = image + rng.normal(0, spec.noise_sigma, image.shape)

    image = np.clip(image, 0, 255).astype(np.uint8)
    if abs(spec.skew_degrees) > 0.1:
        height, width = image.shape
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, spec.skew_degrees, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=242,
        )
    return image


def pearson(x_values: list[float], y_values: list[float]) -> float:
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_denominator = math.sqrt(sum((x_value - x_mean) ** 2 for x_value in x_values))
    y_denominator = math.sqrt(sum((y_value - y_mean) ** 2 for y_value in y_values))
    if x_denominator == 0 or y_denominator == 0:
        return 0.0
    return numerator / (x_denominator * y_denominator)


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end + 2) / 2.0
        for position in range(index, end + 1):
            ranks[indexed[position][0]] = average_rank
        index = end + 1
    return ranks


def spearman(x_values: list[float], y_values: list[float]) -> float:
    return pearson(rank(x_values), rank(y_values))


def run_benchmark(output_dir: Path, count: int, seed: int) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "benchmark_data" / "generated_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = create_base_document()
    rows: list[dict[str, object]] = []
    for offset, spec in enumerate(build_specs(count, seed)):
        image = degrade_document(base, spec, seed + offset)
        image_path = docs_dir / f"{spec.document_id}.png"
        cv2.imwrite(str(image_path), image)
        result = compute_doc_qual_score(image, verbose=False)
        expected_quality = 100 - spec.degradation_level * 20
        rows.append(
            {
                "document_id": spec.document_id,
                "image_path": str(image_path),
                "dataset": "synthetic_docqual_50",
                "degradation_level": spec.degradation_level,
                "expected_quality": expected_quality,
                "doc_qual_score": round(result.ocr_score, 4),
                "passed": result.passed,
                "sharpness": round(result.feature_scores["sharpness"], 4),
                "noise": round(result.feature_scores["noise"], 4),
                "edges": round(result.feature_scores["edges"], 4),
                "skew": round(result.feature_scores["skew"], 4),
                "brightness": round(result.feature_scores["brightness"], 4),
                "ridges": round(result.feature_scores["ridges"], 4),
                "laplacian_variance": round(
                    result.raw_features["laplacian_variance"], 4
                ),
                "noise_std": round(result.raw_features["noise_std"], 4),
                "edge_density": round(result.raw_features["edge_density"], 6),
                "skew_angle": round(result.raw_features["skew_angle"], 4),
                "brightness_mean": round(result.raw_features["brightness_mean"], 4),
                "brightness_std": round(result.raw_features["brightness_std"], 4),
                "ridge_response": round(result.raw_features["ridge_response"], 6),
            }
        )

    scores = [float(row["doc_qual_score"]) for row in rows]
    expected = [float(row["expected_quality"]) for row in rows]
    degradation = [float(row["degradation_level"]) for row in rows]
    hypothesis_rows = [
        {
            "hypothesis": "Doc-Qual increases with expected document quality",
            "metric": "pearson_r",
            "value": round(pearson(scores, expected), 6),
            "n": len(rows),
            "interpretation": "positive values support the hypothesis",
        },
        {
            "hypothesis": "Doc-Qual increases with expected document quality",
            "metric": "spearman_rho",
            "value": round(spearman(scores, expected), 6),
            "n": len(rows),
            "interpretation": "positive values support monotonic ranking",
        },
        {
            "hypothesis": "Doc-Qual decreases as synthetic degradation increases",
            "metric": "pearson_r",
            "value": round(pearson(scores, degradation), 6),
            "n": len(rows),
            "interpretation": "negative values support the hypothesis",
        },
        {
            "hypothesis": "Doc-Qual decreases as synthetic degradation increases",
            "metric": "spearman_rho",
            "value": round(spearman(scores, degradation), 6),
            "n": len(rows),
            "interpretation": "negative values support monotonic ranking",
        },
    ]

    score_csv = output_dir / "doc_qual_50_document_scores.csv"
    hypothesis_csv = output_dir / "doc_qual_hypothesis_summary.csv"
    with score_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with hypothesis_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(hypothesis_rows[0]))
        writer.writeheader()
        writer.writerows(hypothesis_rows)
    return score_csv, hypothesis_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"))
    args = parser.parse_args()
    score_csv, hypothesis_csv = run_benchmark(args.output_dir, args.count, args.seed)
    print(f"Wrote document scores: {score_csv}")
    print(f"Wrote hypothesis summary: {hypothesis_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
