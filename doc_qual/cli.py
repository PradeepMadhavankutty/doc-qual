"""Command line interface for Doc-Qual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doc_qual import __version__

# ── sub-command implementations ──────────────────────────────────────────────


def _cmd_score(args: argparse.Namespace) -> int:
    """Score a single image or PDF."""
    from doc_qual.scorer import compute_doc_qual_score

    image_path = Path(args.image)

    # Route PDF files to the PDF scorer
    if image_path.suffix.lower() == ".pdf":
        try:
            from doc_qual.pdf import score_pdf
        except ImportError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2

        pdf_result = score_pdf(
            image_path,
            threshold=args.threshold,
            engine=args.engine if hasattr(args, "engine") else None,
            dpi=getattr(args, "dpi", 150),
            max_pages=getattr(args, "max_pages", None),
            verbose=args.format == "text",
        )
        if args.format == "json":
            print(json.dumps(pdf_result.to_dict(), indent=2))
        else:
            _print_pdf_text(pdf_result)
        return 0 if pdf_result.passed else 1

    # Grid mode
    if getattr(args, "grid", None):
        try:
            r_str, c_str = args.grid.lower().split("x")
            grid_rows, grid_cols = int(r_str), int(c_str)
        except (ValueError, AttributeError):
            print(
                "[error] --grid must be in ROWSxCOLS format, e.g. 4x4", file=sys.stderr
            )
            return 2

        from doc_qual.grid import score_image_grid

        grid_result = score_image_grid(
            args.image,
            rows=grid_rows,
            cols=grid_cols,
            engine=getattr(args, "engine", None),
            alert_threshold=getattr(args, "alert_threshold", 40.0),
        )
        if args.format == "json":
            print(json.dumps(grid_result.to_dict(), indent=2))
        else:
            _print_grid_text(grid_result, args.threshold)
        return 0 if grid_result.page_score >= args.threshold else 1

    # Standard image scoring
    engine = getattr(args, "engine", None)
    result = compute_doc_qual_score(
        args.image,
        threshold=args.threshold,
        engine=engine,
        verbose=args.format == "text",
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Score an image against a ground-truth file and log the CER."""
    from doc_qual.accuracy.metrics import (
        compute_accuracy,
    )
    from doc_qual.calibration import append_calibration_row, calibration_summary
    from doc_qual.scorer import compute_doc_qual_score

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"[error] Ground-truth file not found: {gt_path}", file=sys.stderr)
        return 2

    ground_truth = gt_path.read_text(encoding="utf-8").strip()
    engine = getattr(args, "engine", "default") or "default"

    # Score the image
    result = compute_doc_qual_score(args.image, verbose=False, engine=None)

    # For CER we need OCR output — if engine is one we can run, do so.
    # Otherwise ask the user to supply hypothesis text via --hypothesis.
    hypothesis_path = getattr(args, "hypothesis", None)
    if hypothesis_path:
        hypothesis = Path(hypothesis_path).read_text(encoding="utf-8").strip()
    else:
        # Fallback: use the image path stem as a placeholder and warn
        print(
            "[warn] No --hypothesis file supplied. "
            "CER will be logged as NaN and excluded from weight fitting.",
            file=sys.stderr,
        )
        hypothesis = ""

    doc_id = Path(args.image).stem
    try:
        acc = compute_accuracy(doc_id, ground_truth, hypothesis)
        cer = acc.cer
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not compute CER: {exc}", file=sys.stderr)
        cer = float("nan")

    cal_path = Path(args.cal_csv) if getattr(args, "cal_csv", None) else None
    kwargs = {} if cal_path is None else {"csv_path": cal_path}
    append_calibration_row(args.image, result.feature_scores, cer, engine, **kwargs)

    summary = calibration_summary(**({"csv_path": cal_path} if cal_path else {}))
    print(
        f"Logged  image={Path(args.image).name}  engine={engine}  "
        f"CER={cer:.4f}  quality={result.ocr_score:.1f}"
    )
    print(f"Dataset now has {summary['total_rows']} rows → {summary['path']}")
    return 0


def _cmd_fit_weights(args: argparse.Namespace) -> int:
    """Fit feature weights from calibration data."""
    from doc_qual.calibration import (
        DEFAULT_CALIBRATION_PATH,
        fit_weights_from_calibration,
    )

    cal_path = (
        Path(args.cal_csv)
        if getattr(args, "cal_csv", None)
        else DEFAULT_CALIBRATION_PATH
    )
    engine = getattr(args, "engine", None)

    try:
        weights = fit_weights_from_calibration(csv_path=cal_path, engine=engine)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    out_path = getattr(args, "output_profile", None)
    if out_path:
        Path(out_path).write_text(json.dumps(weights, indent=2), encoding="utf-8")
        print(f"Weights written to {out_path}")
    else:
        print(json.dumps(weights, indent=2))
    return 0


def _cmd_cal_summary(args: argparse.Namespace) -> int:
    """Print a summary of the calibration dataset."""
    from doc_qual.calibration import DEFAULT_CALIBRATION_PATH, calibration_summary

    cal_path = (
        Path(args.cal_csv)
        if getattr(args, "cal_csv", None)
        else DEFAULT_CALIBRATION_PATH
    )
    summary = calibration_summary(csv_path=cal_path)
    print(json.dumps(summary, indent=2))
    return 0


# ── text formatters ───────────────────────────────────────────────────────────


def _print_grid_text(grid_result: object, threshold: float) -> None:
    from doc_qual.grid import GridResult

    if not isinstance(grid_result, GridResult):
        return
    print(f"\nGrid Quality Score ({grid_result.grid_rows}×{grid_result.grid_cols})")
    print(
        f"  Page score : {grid_result.page_score:.1f}  "
        f"({'PASS' if grid_result.page_score >= threshold else 'FAIL'})"
    )
    if grid_result.worst_cell:
        wc = grid_result.worst_cell
        print(f"  Worst cell : row={wc.row} col={wc.col}  score={wc.score:.1f}")
    if grid_result.worst_region_alert:
        print(f"  ⚠ ALERT: worst cell below {grid_result.alert_threshold:.0f}")
    print("\n  Heatmap (░▒▓█ = low→high quality):")
    for line in grid_result.ascii_heatmap().splitlines():
        print(f"    {line}")


def _print_pdf_text(pdf_result: object) -> None:
    from doc_qual.pdf import PDFQualityResult

    if not isinstance(pdf_result, PDFQualityResult):
        return
    print(f"\nPDF Quality Report: {pdf_result.path}")
    print(f"  Pages        : {pdf_result.page_count}")
    print(
        f"  Summary score: {pdf_result.summary_score:.1f}  "
        f"({'PASS' if pdf_result.passed else 'FAIL'})"
    )
    print(f"  Worst page   : {pdf_result.worst_page}")
    print()
    print(f"  {'Page':>4}  {'Score':>6}  {'Pass':>5}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*5}")
    for p in pdf_result.pages:
        print(
            f"  {p.page_number:>4}  {p.ocr_score:>6.1f}  {'✓' if p.passed else '✗':>5}"
        )


# ── parser builder ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-qual",
        description="Score document image quality for OCR readiness.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    # ── score (default) ───────────────────────────────────────────────────
    score_p = sub.add_parser("score", help="Score a document image or PDF.")
    score_p.add_argument("image", help="Path to the image or PDF to score.")
    score_p.add_argument(
        "--format", choices=("text", "json"), default="text", help="Output format."
    )
    score_p.add_argument(
        "--threshold", type=float, default=60.0, help="Minimum passing score."
    )
    score_p.add_argument(
        "--engine",
        choices=("tesseract", "textract", "azure", "paddleocr"),
        default=None,
        help="Engine-specific weight profile.",
    )
    score_p.add_argument(
        "--grid",
        metavar="ROWSxCOLS",
        default=None,
        help="Enable per-region grid scoring (e.g. 4x4).",
    )
    score_p.add_argument(
        "--alert-threshold",
        type=float,
        default=40.0,
        dest="alert_threshold",
        help="Worst-cell alert threshold when using --grid.",
    )
    score_p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Render DPI for PDF input (default: 150).",
    )
    score_p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        dest="max_pages",
        help="Limit number of pages scored for PDF input.",
    )

    # ── calibrate ─────────────────────────────────────────────────────────
    cal_p = sub.add_parser(
        "calibrate",
        help="Score an image and log (features, CER) for weight calibration.",
    )
    cal_p.add_argument("image", help="Path to the image to score.")
    cal_p.add_argument(
        "--ground-truth",
        required=True,
        dest="ground_truth",
        metavar="FILE",
        help="Plain-text file containing the known correct text for this image.",
    )
    cal_p.add_argument(
        "--hypothesis",
        default=None,
        metavar="FILE",
        help="Plain-text file containing OCR output for CER calculation.",
    )
    cal_p.add_argument(
        "--engine",
        default="default",
        help="Engine name to tag this calibration row.",
    )
    cal_p.add_argument(
        "--cal-csv",
        default=None,
        dest="cal_csv",
        metavar="PATH",
        help="Override calibration CSV path.",
    )

    # ── fit-weights ───────────────────────────────────────────────────────
    fit_p = sub.add_parser(
        "fit-weights",
        help="Fit feature weights from accumulated calibration data.",
    )
    fit_p.add_argument(
        "--engine",
        default=None,
        help="Only use calibration rows for this engine.",
    )
    fit_p.add_argument(
        "--cal-csv",
        default=None,
        dest="cal_csv",
        metavar="PATH",
        help="Override calibration CSV path.",
    )
    fit_p.add_argument(
        "--output-profile",
        default=None,
        dest="output_profile",
        metavar="PATH",
        help="Write fitted weights to a JSON profile file.",
    )

    # ── cal-summary ───────────────────────────────────────────────────────
    sum_p = sub.add_parser("cal-summary", help="Print calibration dataset summary.")
    sum_p.add_argument(
        "--cal-csv",
        default=None,
        dest="cal_csv",
        metavar="PATH",
        help="Override calibration CSV path.",
    )

    return parser


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    effective = list(argv) if argv is not None else sys.argv[1:]

    # Backward compatibility: ``doc-qual image.jpg [opts]`` (no subcommand).
    # If the first non-flag token is not a known subcommand, prepend 'score'.
    _subcmds = {
        "score",
        "calibrate",
        "fit-weights",
        "cal-summary",
        "--help",
        "-h",
        "--version",
    }
    if effective and not effective[0].startswith("-") and effective[0] not in _subcmds:
        effective = ["score"] + effective

    parser = build_parser()
    args = parser.parse_args(effective)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "score": _cmd_score,
        "calibrate": _cmd_calibrate,
        "fit-weights": _cmd_fit_weights,
        "cal-summary": _cmd_cal_summary,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
