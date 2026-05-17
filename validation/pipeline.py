"""Main validation pipeline — orchestrates all seven phases.

Usage::

    from validation import ValidationConfig, ValidationPipeline

    cfg = ValidationConfig(n_documents=54, ocr_provider="mock")
    pipeline = ValidationPipeline(cfg)
    report = pipeline.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validation.accuracy.metrics import AccuracyResult, compute_accuracy
from validation.config import ValidationConfig
from validation.dataset.generator import DocumentGenerator
from validation.dataset.manifest import DatasetManifest
from validation.ocr.base import OCRResult
from validation.ocr.mock_provider import MockOCRProvider
from validation.ocr.registry import get_provider
from validation.quality.indexer import QualityIndexResult, compute_quality_index
from validation.reports.artifacts import ArtifactWriter
from validation.reports.plots import PlotGenerator
from validation.stats.hypothesis import (
    HypothesisResult,
    validate_quality_accuracy_hypothesis,
)


@dataclass
class ValidationReport:
    config: ValidationConfig
    manifest_summary: dict[str, Any] = field(default_factory=dict)
    ocr_results: list[OCRResult] = field(default_factory=list)
    quality_results: list[QualityIndexResult] = field(default_factory=list)
    accuracy_results: list[AccuracyResult] = field(default_factory=list)
    hypothesis_cer: HypothesisResult | None = None
    hypothesis_wer: HypothesisResult | None = None
    artifact_paths: list[Path] = field(default_factory=list)
    elapsed_s: float = 0.0

    # ── computed summaries ─────────────────────────────────────────────────

    def quality_stats(self) -> dict[str, float]:
        scores = [r.ocr_score for r in self.quality_results]
        if not scores:
            return {}
        n = len(scores)
        mu = sum(scores) / n
        variance = sum((s - mu) ** 2 for s in scores) / max(n - 1, 1)
        std = variance**0.5
        return {
            "mean": mu,
            "std": std,
            "min": min(scores),
            "max": max(scores),
            "pass_rate": sum(1 for r in self.quality_results if r.passed) / n,
        }

    def accuracy_stats(self) -> dict[str, float]:
        if not self.accuracy_results:
            return {}
        cers = [r.cer for r in self.accuracy_results]
        wers = [r.wer for r in self.accuracy_results]
        exact = [r.exact_match for r in self.accuracy_results]

        def _stats(xs: list[float]) -> tuple[float, float, float, float]:
            mu = sum(xs) / len(xs)
            std = (sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
            return mu, std, min(xs), max(xs)

        c_mu, c_std, c_min, c_max = _stats(cers)
        w_mu, w_std, w_min, w_max = _stats(wers)
        return {
            "cer_mean": c_mu,
            "cer_std": c_std,
            "cer_min": c_min,
            "cer_max": c_max,
            "wer_mean": w_mu,
            "wer_std": w_std,
            "wer_min": w_min,
            "wer_max": w_max,
            "exact_match_rate": sum(exact) / len(exact),
        }


class ValidationPipeline:
    """Runs the full seven-phase validation workflow."""

    def __init__(self, cfg: ValidationConfig) -> None:
        self._cfg = cfg

    def run(self) -> ValidationReport:
        t0 = time.perf_counter()
        cfg = self._cfg
        cfg.makedirs()

        report = ValidationReport(config=cfg)
        writer = ArtifactWriter(cfg.output_dir)
        plotter = PlotGenerator(cfg.plots_dir)

        # ── Phase 1: Dataset generation ───────────────────────────────────
        self._log("Phase 1 — Generating synthetic dataset …")
        generator = DocumentGenerator(cfg)
        manifest: DatasetManifest = generator.generate(cfg.dataset_dir)
        report.manifest_summary = manifest.summary()

        manifest.save(cfg.dataset_dir / "manifest.json")
        paths = writer.write_dataset_summary(
            report.manifest_summary,
            [r.to_dict() for r in manifest],
        )
        report.artifact_paths.extend(paths)
        self._log(
            f"  Generated {len(manifest)} documents across "
            f"{len(report.manifest_summary['document_types'])} types."
        )

        # ── Phase 2: OCR extraction ───────────────────────────────────────
        self._log(f"Phase 2 — Running OCR with provider '{cfg.ocr_provider}' …")
        provider = get_provider(cfg.ocr_provider, seed=cfg.seed)

        # Register ground truth for mock provider
        if isinstance(provider, MockOCRProvider):
            for doc in manifest:
                provider.register_document(
                    doc.document_id, doc.ground_truth, doc.quality_condition
                )

        ocr_results: list[OCRResult] = []
        for doc in manifest:
            result = provider.run(Path(doc.image_path), doc.document_id)
            ocr_results.append(result)
            if not result.success:
                self._log(f"  [WARN] OCR failed for {doc.document_id}: {result.error}")

        report.ocr_results = ocr_results
        ocr_success = sum(1 for r in ocr_results if r.success)
        self._log(f"  OCR success: {ocr_success}/{len(ocr_results)}")

        paths = writer.write_ocr_results([r.to_dict() for r in ocr_results])
        report.artifact_paths.extend(paths)

        # ── Phase 3: Quality index ────────────────────────────────────────
        self._log("Phase 3 — Computing OCR Quality Index …")
        quality_results: list[QualityIndexResult] = []
        for doc in manifest:
            try:
                qi = compute_quality_index(
                    Path(doc.image_path), doc.document_id, cfg.doc_qual_threshold
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"  [WARN] Quality index failed for {doc.document_id}: {exc}")
                qi = QualityIndexResult(
                    document_id=doc.document_id,
                    ocr_score=0.0,
                    passed=False,
                    threshold=cfg.doc_qual_threshold,
                    feature_scores={},
                    raw_features={},
                    recommendations=[f"Error: {exc}"],
                )
            quality_results.append(qi)

        report.quality_results = quality_results
        paths = writer.write_quality_index([r.to_dict() for r in quality_results])
        report.artifact_paths.extend(paths)
        q_stats = report.quality_stats()
        self._log(
            f"  Quality mean={q_stats.get('mean', 0):.1f}  "
            f"pass_rate={q_stats.get('pass_rate', 0):.1%}"
        )

        # ── Phase 4: Accuracy validation ──────────────────────────────────
        self._log("Phase 4 — Computing accuracy against ground truth …")
        doc_map = {doc.document_id: doc for doc in manifest}
        ocr_map = {r.document_id: r for r in ocr_results}

        accuracy_results: list[AccuracyResult] = []
        for doc_id, doc in doc_map.items():
            ocr = ocr_map.get(doc_id)
            hypothesis_text = ocr.extracted_text if (ocr and ocr.success) else ""
            acc = compute_accuracy(doc_id, doc.ground_truth, hypothesis_text)
            accuracy_results.append(acc)

        report.accuracy_results = accuracy_results
        paths = writer.write_accuracy_results([r.to_dict() for r in accuracy_results])
        report.artifact_paths.extend(paths)
        a_stats = report.accuracy_stats()
        self._log(
            f"  Mean CER={a_stats.get('cer_mean', 0):.4f}  "
            f"Mean WER={a_stats.get('wer_mean', 0):.4f}  "
            f"Exact match={a_stats.get('exact_match_rate', 0):.1%}"
        )

        # ── Phase 5: Hypothesis testing ───────────────────────────────────
        self._log("Phase 5 — Running hypothesis tests …")
        quality_scores = [r.ocr_score for r in quality_results]
        cer_values = [r.cer for r in accuracy_results]
        wer_values = [r.wer for r in accuracy_results]
        condition_labels = [
            doc_map[r.document_id].quality_condition for r in accuracy_results
        ]

        h_cer = validate_quality_accuracy_hypothesis(
            quality_scores, cer_values, cfg.confidence_level, condition_labels
        )
        h_wer = validate_quality_accuracy_hypothesis(
            quality_scores, wer_values, cfg.confidence_level, condition_labels
        )
        report.hypothesis_cer = h_cer
        report.hypothesis_wer = h_wer

        paths = writer.write_hypothesis([h_cer.to_dict(), h_wer.to_dict()])
        report.artifact_paths.extend(paths)
        self._log(
            f"  CER hypothesis: r={h_cer.pearson_r:.3f}  "
            f"H0={'rejected' if h_cer.reject_h0 else 'not rejected'}"
        )

        # ── Phase 6: Research analysis ────────────────────────────────────
        self._log("Phase 6 — Generating research artifacts …")
        research_tables = self._build_research_tables(
            manifest, quality_results, accuracy_results, ocr_results
        )
        paths = writer.write_research_tables(research_tables)
        report.artifact_paths.extend(paths)

        # Markdown report + paper assets
        q_stats = report.quality_stats()
        a_stats = report.accuracy_stats()
        md_path = writer.write_markdown_report(
            report.manifest_summary, h_cer.to_dict(), a_stats, q_stats
        )
        paper_paths = writer.write_paper_assets(
            report.manifest_summary, h_cer.to_dict(), a_stats, q_stats
        )
        report.artifact_paths.append(md_path)
        report.artifact_paths.extend(paper_paths)

        # ── Phase 7: Plots ────────────────────────────────────────────────
        self._log("Phase 7 — Generating plots …")
        try:
            plot_paths = self._generate_plots(
                plotter,
                quality_scores,
                cer_values,
                wer_values,
                condition_labels,
                h_cer.pearson_r,
                quality_results,
                accuracy_results,
            )
            report.artifact_paths.extend(plot_paths)
            self._log(f"  Generated {len(plot_paths)} plots.")
        except ImportError as exc:
            self._log(f"  [SKIP] Plots skipped (matplotlib not installed): {exc}")

        report.elapsed_s = time.perf_counter() - t0
        self._log(f"Pipeline complete in {report.elapsed_s:.1f}s.")
        return report

    # ── internal helpers ───────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self._cfg.verbose:
            print(msg)

    @staticmethod
    def _build_research_tables(
        manifest: DatasetManifest,
        quality_results: list[QualityIndexResult],
        accuracy_results: list[AccuracyResult],
        ocr_results: list[OCRResult],
    ) -> dict[str, list[dict[str, Any]]]:
        {doc.document_id: doc for doc in manifest}
        qi_map = {r.document_id: r for r in quality_results}
        acc_map = {r.document_id: r for r in accuracy_results}
        ocr_map = {r.document_id: r for r in ocr_results}

        # Table 1 — per-document full record
        per_doc: list[dict[str, Any]] = []
        for doc in manifest:
            qi = qi_map.get(doc.document_id)
            acc = acc_map.get(doc.document_id)
            ocr = ocr_map.get(doc.document_id)
            per_doc.append(
                {
                    "document_id": doc.document_id,
                    "document_type": doc.document_type,
                    "quality_condition": doc.quality_condition,
                    "difficulty_level": doc.difficulty_level,
                    "quality_score": round(qi.ocr_score, 4) if qi else None,
                    "quality_passed": qi.passed if qi else None,
                    "cer": round(acc.cer, 6) if acc else None,
                    "wer": round(acc.wer, 6) if acc else None,
                    "exact_match": acc.exact_match if acc else None,
                    "ocr_confidence": round(ocr.confidence, 4) if ocr else None,
                    "ocr_latency_ms": round(ocr.latency_ms, 1) if ocr else None,
                }
            )

        # Table 2 — accuracy by condition
        from collections import defaultdict

        condition_groups: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"quality": [], "cer": [], "wer": []}
        )
        for doc in manifest:
            qi = qi_map.get(doc.document_id)
            acc = acc_map.get(doc.document_id)
            if qi and acc:
                condition_groups[doc.quality_condition]["quality"].append(qi.ocr_score)
                condition_groups[doc.quality_condition]["cer"].append(acc.cer)
                condition_groups[doc.quality_condition]["wer"].append(acc.wer)

        by_condition: list[dict[str, Any]] = []
        for cond, vals in sorted(condition_groups.items()):
            q = vals["quality"]
            c = vals["cer"]
            w = vals["wer"]
            by_condition.append(
                {
                    "condition": cond,
                    "n": len(q),
                    "mean_quality": round(sum(q) / len(q), 2) if q else 0,
                    "mean_cer": round(sum(c) / len(c), 4) if c else 0,
                    "mean_wer": round(sum(w) / len(w), 4) if w else 0,
                }
            )

        # Table 3 — accuracy by document type
        type_groups: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"quality": [], "cer": [], "wer": []}
        )
        for doc in manifest:
            qi = qi_map.get(doc.document_id)
            acc = acc_map.get(doc.document_id)
            if qi and acc:
                type_groups[doc.document_type]["quality"].append(qi.ocr_score)
                type_groups[doc.document_type]["cer"].append(acc.cer)
                type_groups[doc.document_type]["wer"].append(acc.wer)

        by_type: list[dict[str, Any]] = []
        for dtype, vals in sorted(type_groups.items()):
            q = vals["quality"]
            c = vals["cer"]
            w = vals["wer"]
            by_type.append(
                {
                    "document_type": dtype,
                    "n": len(q),
                    "mean_quality": round(sum(q) / len(q), 2) if q else 0,
                    "mean_cer": round(sum(c) / len(c), 4) if c else 0,
                    "mean_wer": round(sum(w) / len(w), 4) if w else 0,
                }
            )

        return {
            "per_document_results": per_doc,
            "accuracy_by_condition": by_condition,
            "accuracy_by_document_type": by_type,
        }

    @staticmethod
    def _generate_plots(
        plotter: PlotGenerator,
        quality_scores: list[float],
        cer_values: list[float],
        wer_values: list[float],
        condition_labels: list[str],
        pearson_r: float,
        quality_results: list[QualityIndexResult],
        accuracy_results: list[AccuracyResult],
    ) -> list[Path]:
        paths: list[Path] = []

        paths.append(
            plotter.quality_vs_cer(
                quality_scores, cer_values, condition_labels, pearson_r
            )
        )
        paths.append(
            plotter.quality_vs_wer(quality_scores, wer_values, condition_labels)
        )
        paths.append(plotter.quality_distribution(quality_scores))
        paths.append(plotter.error_rate_distribution(cer_values, wer_values))
        paths.append(plotter.quality_by_condition(quality_scores, condition_labels))
        paths.append(plotter.wer_by_condition(wer_values, condition_labels))

        # Correlation heatmap using feature scores from first result with data
        if quality_results and quality_results[0].feature_scores:
            feature_keys = list(quality_results[0].feature_scores.keys())
            feature_data: dict[str, list[float]] = {k: [] for k in feature_keys}
            feature_data["quality_index"] = []
            feature_data["cer"] = []
            feature_data["wer"] = []
            acc_map = {r.document_id: r for r in accuracy_results}
            for qi in quality_results:
                acc = acc_map.get(qi.document_id)
                if acc:
                    for k in feature_keys:
                        feature_data[k].append(qi.feature_scores.get(k, 0.0))
                    feature_data["quality_index"].append(qi.ocr_score)
                    feature_data["cer"].append(acc.cer)
                    feature_data["wer"].append(acc.wer)
            if all(len(v) >= 2 for v in feature_data.values()):
                paths.append(plotter.correlation_heatmap(feature_data))

        # Degradation sensitivity
        unique_conds = sorted(set(condition_labels))
        mean_q_per_cond: list[float] = []
        mean_w_per_cond: list[float] = []
        for cond in unique_conds:
            qs = [q for q, c in zip(quality_scores, condition_labels) if c == cond]
            ws = [w for w, c in zip(wer_values, condition_labels) if c == cond]
            mean_q_per_cond.append(sum(qs) / len(qs) if qs else 0.0)
            mean_w_per_cond.append(sum(ws) / len(ws) if ws else 0.0)

        paths.append(
            plotter.degradation_sensitivity(
                unique_conds, mean_q_per_cond, mean_w_per_cond
            )
        )

        return paths
