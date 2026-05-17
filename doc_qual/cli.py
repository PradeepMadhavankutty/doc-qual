"""Command line interface for Doc-Qual."""

from __future__ import annotations

import argparse
import json
import sys

from doc_qual import __version__
from doc_qual.scorer import compute_doc_qual_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-qual",
        description="Score document image quality for OCR readiness.",
    )
    parser.add_argument("image", help="Path to the image to score")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Minimum passing score. Exits 1 when the score is lower.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = compute_doc_qual_score(
        args.image,
        threshold=args.threshold,
        verbose=args.format == "text",
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
