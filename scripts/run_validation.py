"""CLI entry point for the OCR validation framework.

Examples::

    # Quick run with mock OCR (no external dependencies)
    python scripts/run_validation.py --n 54 --provider mock --verbose

    # Full run with Tesseract
    python scripts/run_validation.py --n 100 --provider tesseract --verbose

    # Custom output directory
    python scripts/run_validation.py --output /tmp/my_validation --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `validation` and `doc_qual` are importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from validation.config import ValidationConfig  # noqa: E402
from validation.pipeline import ValidationPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_validation",
        description="OCR Quality Validation — doc_qual functional test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Validation"),
        metavar="DIR",
        help="Root output directory (default: Validation/)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=54,
        metavar="N",
        help="Number of synthetic documents to generate (default: 54)",
    )
    p.add_argument(
        "--provider",
        choices=("mock", "tesseract", "easyocr"),
        default="mock",
        help="OCR provider to use (default: mock)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="doc_qual pass threshold (default: 60.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for hypothesis tests (default: 0.95)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stdout",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a JSON summary to stdout after completion",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = ValidationConfig(
        output_dir=args.output,
        n_documents=args.n,
        ocr_provider=args.provider,
        doc_qual_threshold=args.threshold,
        seed=args.seed,
        confidence_level=args.confidence,
        verbose=args.verbose,
    )

    pipeline = ValidationPipeline(cfg)
    report = pipeline.run()

    if args.summary or args.verbose:
        q = report.quality_stats()
        a = report.accuracy_stats()
        h = report.hypothesis_cer
        summary = {
            "documents": len(report.ocr_results),
            "ocr_provider": cfg.ocr_provider,
            "mean_quality": round(q.get("mean", 0), 2),
            "pass_rate": round(q.get("pass_rate", 0), 4),
            "mean_cer": round(a.get("cer_mean", 0), 4),
            "mean_wer": round(a.get("wer_mean", 0), 4),
            "pearson_r": round(h.pearson_r, 4) if h else None,
            "reject_h0": h.reject_h0 if h else None,
            "elapsed_s": round(report.elapsed_s, 1),
            "artifacts": len(report.artifact_paths),
            "output_dir": str(cfg.output_dir.resolve()),
        }
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
