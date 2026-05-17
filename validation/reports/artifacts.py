"""Artifact writer — CSV, JSON, and Markdown report generation."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Any


class ArtifactWriter:
    """Writes structured validation results to multiple output formats."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    # ── generic helpers ────────────────────────────────────────────────────

    def write_json(self, filename: str, data: Any, subdir: str = "") -> Path:
        path = (self._base / subdir / filename) if subdir else self._base / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def write_csv(
        self,
        filename: str,
        rows: list[dict[str, Any]],
        subdir: str = "",
    ) -> Path:
        if not rows:
            return self._base / filename
        path = (self._base / subdir / filename) if subdir else self._base / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    # ── dataset manifest ──────────────────────────────────────────────────

    def write_dataset_summary(
        self, summary: dict[str, Any], manifest_rows: list[dict[str, Any]]
    ) -> list[Path]:
        paths = [
            self.write_json("dataset_summary.json", summary, "Dataset"),
            self.write_csv("document_manifest.csv", manifest_rows, "Dataset"),
        ]
        return paths

    # ── OCR results ───────────────────────────────────────────────────────

    def write_ocr_results(self, rows: list[dict[str, Any]]) -> list[Path]:
        return [
            self.write_json("ocr_results.json", rows, "OCR_Output"),
            self.write_csv("ocr_results.csv", rows, "OCR_Output"),
        ]

    # ── quality index ─────────────────────────────────────────────────────

    def write_quality_index(self, rows: list[dict[str, Any]]) -> list[Path]:
        return [
            self.write_json("quality_index.json", rows, "Quality_Index"),
            self.write_csv(
                "quality_index.csv",
                [_flatten(r) for r in rows],
                "Quality_Index",
            ),
        ]

    # ── accuracy results ──────────────────────────────────────────────────

    def write_accuracy_results(self, rows: list[dict[str, Any]]) -> list[Path]:
        return [
            self.write_json("accuracy_results.json", rows, "Ground_Truth_Analysis"),
            self.write_csv("accuracy_results.csv", rows, "Ground_Truth_Analysis"),
        ]

    # ── hypothesis results ────────────────────────────────────────────────

    def write_hypothesis(self, results: list[dict[str, Any]]) -> list[Path]:
        return [
            self.write_json("hypothesis_results.json", results, "Hypothesis_Testing"),
            self.write_csv("hypothesis_summary.csv", results, "Hypothesis_Testing"),
        ]

    # ── research artifacts ────────────────────────────────────────────────

    def write_research_tables(
        self, tables: dict[str, list[dict[str, Any]]]
    ) -> list[Path]:
        paths: list[Path] = []
        for name, rows in tables.items():
            paths.append(self.write_csv(f"{name}.csv", rows, "Research_Artifacts"))
        return paths

    # ── markdown reports ──────────────────────────────────────────────────

    def write_markdown_report(
        self,
        summary: dict[str, Any],
        hypothesis_result: dict[str, Any],
        accuracy_stats: dict[str, Any],
        quality_stats: dict[str, Any],
    ) -> Path:
        md = _build_markdown_report(
            summary, hypothesis_result, accuracy_stats, quality_stats
        )
        path = self._base / "Reports" / "validation_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        return path

    def write_paper_assets(
        self,
        summary: dict[str, Any],
        hypothesis_result: dict[str, Any],
        accuracy_stats: dict[str, Any],
        quality_stats: dict[str, Any],
    ) -> list[Path]:
        paper_md = _build_paper_assets(
            summary, hypothesis_result, accuracy_stats, quality_stats
        )
        path = self._base / "Paper_Assets" / "paper_ready_summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(paper_md, encoding="utf-8")
        return [path]


# ── helpers ────────────────────────────────────────────────────────────────


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict for CSV output."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = "; ".join(str(i) for i in v)
        else:
            out[key] = v
    return out


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def _build_markdown_report(
    summary: dict[str, Any],
    hyp: dict[str, Any],
    acc: dict[str, Any],
    qual: dict[str, Any],
) -> str:
    reject_label = "**REJECTED** ✓" if hyp.get("reject_h0") else "Not rejected"
    return textwrap.dedent(f"""\
    # OCR Quality Validation Report

    Generated by the doc_qual validation framework.

    ## 1. Dataset Summary

    | Metric | Value |
    |--------|-------|
    | Total documents | {summary.get("total_documents", "N/A")} |
    | Document types | {len(summary.get("document_types", {}))} |
    | Quality conditions | {len(summary.get("quality_conditions", {}))} |
    | Difficulty levels | {len(summary.get("difficulty_levels", {}))} |

    ### Document Type Distribution

    | Type | Count |
    |------|-------|
    {_table_rows(summary.get("document_types", {}))}

    ### Quality Condition Distribution

    | Condition | Count |
    |-----------|-------|
    {_table_rows(summary.get("quality_conditions", {}))}

    ---

    ## 2. OCR Quality Index Summary

    | Metric | Value |
    |--------|-------|
    | Mean quality score | {qual.get("mean", 0):.2f} |
    | Std dev | {qual.get("std", 0):.2f} |
    | Min | {qual.get("min", 0):.2f} |
    | Max | {qual.get("max", 0):.2f} |
    | Pass rate (≥60) | {qual.get("pass_rate", 0):.1%} |

    ---

    ## 3. Accuracy Results

    | Metric | Mean | Std | Min | Max |
    |--------|------|-----|-----|-----|
    | CER | {acc.get("cer_mean", 0):.4f} | {acc.get("cer_std", 0):.4f} | {acc.get("cer_min", 0):.4f} | {acc.get("cer_max", 0):.4f} |
    | WER | {acc.get("wer_mean", 0):.4f} | {acc.get("wer_std", 0):.4f} | {acc.get("wer_min", 0):.4f} | {acc.get("wer_max", 0):.4f} |
    | Exact match | {acc.get("exact_match_rate", 0):.1%} | — | — | — |

    ---

    ## 4. Hypothesis Validation

    > **H1**: Higher OCR Quality Index leads to lower OCR extraction error rate.
    >
    > **H0**: OCR Quality Index has no statistically significant relationship.

    | Test | Statistic | p-value | Decision |
    |------|-----------|---------|----------|
    | Pearson r | {hyp.get("pearson_r", 0):.4f} | {_fmt_p(hyp.get("pearson_p", 1.0))} | H0 {reject_label} |
    | Spearman ρ | {hyp.get("spearman_rho", 0):.4f} | {_fmt_p(hyp.get("spearman_p", 1.0))} | — |
    | ANOVA F | {hyp.get("anova_f", 0):.4f} | {_fmt_p(hyp.get("anova_p", 1.0))} | — |

    **Linear regression**: error_rate = {hyp.get("regression_slope", 0):.4f} × quality + {hyp.get("regression_intercept", 0):.4f}  (R² = {hyp.get("regression_r2", 0):.3f})

    **95% CI for r**: [{hyp.get("ci_lower", 0):.3f}, {hyp.get("ci_upper", 0):.3f}]

    {hyp.get("interpretation", "")}

    ---

    ## 5. Recommendations

    {_notes_list(hyp.get("notes", []))}
    """)


def _build_paper_assets(
    summary: dict[str, Any],
    hyp: dict[str, Any],
    acc: dict[str, Any],
    qual: dict[str, Any],
) -> str:
    return textwrap.dedent(f"""\
    # Paper-Ready Summary — OCR Quality Index Validation

    ## Abstract

    We present a functional validation of the Doc-Qual OCR Quality Index against
    ground-truth OCR extraction accuracy across {summary.get("total_documents", "N/A")}
    synthetic document images spanning {len(summary.get("document_types", {}))} document
    types and {len(summary.get("quality_conditions", {}))} quality degradation conditions.
    Pearson r = {hyp.get("pearson_r", 0):.3f} (p {_fmt_p(hyp.get("pearson_p", 1.0))})
    confirms a statistically {"significant" if hyp.get("reject_h0") else "insignificant"}
    negative correlation between quality score and character error rate,
    {'supporting' if hyp.get("reject_h0") else 'not supporting'} H1.

    ## Experimental Design

    - **Dataset**: {summary.get("total_documents", "N/A")} synthetic documents,
      balanced across {len(summary.get("document_types", {}))} types and
      {len(summary.get("quality_conditions", {}))} quality conditions.
    - **Quality scoring**: Doc-Qual composite index (sharpness, noise, edges,
      skew, brightness, ridges).
    - **OCR engine**: Mock provider with condition-calibrated word-error injection.
    - **Accuracy metrics**: Character Error Rate (CER) and Word Error Rate (WER)
      against known ground truth.
    - **Statistical tests**: Pearson r, Spearman ρ, linear regression, one-way ANOVA.

    ## Key Findings

    | Finding | Value |
    |---------|-------|
    | Mean quality score | {qual.get("mean", 0):.2f} / 100 |
    | Mean CER | {acc.get("cer_mean", 0):.4f} |
    | Mean WER | {acc.get("wer_mean", 0):.4f} |
    | Pearson r (quality vs CER) | {hyp.get("pearson_r", 0):.4f} |
    | Linear regression R² | {hyp.get("regression_r2", 0):.3f} |
    | H0 rejected | {hyp.get("reject_h0", False)} |

    ## Statistical Tables

    **Table 1 — Accuracy by quality condition** (see accuracy_by_condition.csv)

    **Table 2 — Hypothesis test summary** (see hypothesis_results.json)

    ## Limitations

    - Synthetic dataset; real-world documents may exhibit different degradation patterns.
    - Mock OCR errors are stochastic; replace with a real OCR engine for production use.
    - Quality scoring limited to image-level features; layout and semantic metrics
      require additional tooling.

    ## Future Work

    - Validate with real OCR engines (Tesseract, PaddleOCR, Azure OCR).
    - Extend to multilingual documents.
    - Add layout-aware quality metrics (table detection, paragraph continuity).
    - Publish annotated benchmark dataset for reproducibility.
    """)


def _table_rows(d: dict[str, int]) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in sorted(d.items()))


def _notes_list(notes: list[str]) -> str:
    return "\n".join(f"- {n}" for n in notes)
