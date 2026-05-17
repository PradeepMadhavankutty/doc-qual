"""Matplotlib-based plot generation for the OCR validation framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _ensure_matplotlib() -> None:
    """Verify matplotlib is importable and set the Agg backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot generation. "
            "Install with: pip install matplotlib"
        ) from exc


class PlotGenerator:
    """Generate and save publication-quality validation charts."""

    def __init__(self, output_dir: Path, dpi: int = 150) -> None:
        self._dir = output_dir
        self._dpi = dpi
        self._dir.mkdir(parents=True, exist_ok=True)

    def _save(self, plt: Any, name: str) -> Path:
        path = self._dir / name
        plt.savefig(path, dpi=self._dpi, bbox_inches="tight")
        plt.close()
        return path

    # ── scatter: quality vs error rate ────────────────────────────────────

    def quality_vs_cer(
        self,
        quality_scores: list[float],
        cer_values: list[float],
        condition_labels: list[str] | None = None,
        pearson_r: float | None = None,
    ) -> Path:
        _ensure_matplotlib()
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt2
        import numpy as np

        fig, ax = plt2.subplots(figsize=(8, 5))

        if condition_labels:
            unique = sorted(set(condition_labels))
            colors = cm.tab10([i / max(len(unique) - 1, 1) for i in range(len(unique))])
            color_map = dict(zip(unique, colors))
            for cond in unique:
                xs = [q for q, c in zip(quality_scores, condition_labels) if c == cond]
                ys = [e for e, c in zip(cer_values, condition_labels) if c == cond]
                ax.scatter(xs, ys, label=cond, color=color_map[cond], alpha=0.75, s=55)
            ax.legend(title="Condition", fontsize=8, loc="upper right")
        else:
            ax.scatter(quality_scores, cer_values, alpha=0.65, s=55, color="steelblue")

        # regression line
        if len(quality_scores) >= 2:
            xs_arr = np.array(quality_scores)
            ys_arr = np.array(cer_values)
            m = np.polyfit(xs_arr, ys_arr, 1)
            x_line = np.linspace(min(xs_arr), max(xs_arr), 200)
            ax.plot(x_line, np.polyval(m, x_line), "r--", lw=1.5, label="Regression")

        title = "OCR Quality Index vs. Character Error Rate"
        if pearson_r is not None:
            title += f"\nPearson r = {pearson_r:.3f}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("OCR Quality Index (0–100)")
        ax.set_ylabel("Character Error Rate (CER)")
        ax.grid(True, alpha=0.3)
        return self._save(plt2, "quality_vs_cer.png")

    def quality_vs_wer(
        self,
        quality_scores: list[float],
        wer_values: list[float],
        condition_labels: list[str] | None = None,
        pearson_r: float | None = None,
    ) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2
        import numpy as np

        fig, ax = plt2.subplots(figsize=(8, 5))
        ax.scatter(quality_scores, wer_values, alpha=0.65, s=55, color="darkorange")

        if len(quality_scores) >= 2:
            xs_arr = np.array(quality_scores)
            ys_arr = np.array(wer_values)
            m = np.polyfit(xs_arr, ys_arr, 1)
            x_line = np.linspace(min(xs_arr), max(xs_arr), 200)
            ax.plot(x_line, np.polyval(m, x_line), "b--", lw=1.5, label="Regression")

        title = "OCR Quality Index vs. Word Error Rate"
        if pearson_r is not None:
            title += f"\nPearson r = {pearson_r:.3f}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("OCR Quality Index (0–100)")
        ax.set_ylabel("Word Error Rate (WER)")
        ax.grid(True, alpha=0.3)
        return self._save(plt2, "quality_vs_wer.png")

    # ── distribution histograms ───────────────────────────────────────────

    def quality_distribution(self, quality_scores: list[float]) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2
        import numpy as np

        fig, ax = plt2.subplots(figsize=(7, 4))
        ax.hist(
            quality_scores, bins=20, color="steelblue", edgecolor="white", alpha=0.85
        )
        ax.axvline(60, color="red", linestyle="--", lw=1.5, label="Threshold (60)")
        ax.axvline(
            np.mean(quality_scores),
            color="green",
            linestyle="-",
            lw=1.5,
            label=f"Mean = {np.mean(quality_scores):.1f}",
        )
        ax.set_title("OCR Quality Index Distribution")
        ax.set_xlabel("Quality Score (0–100)")
        ax.set_ylabel("Document Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save(plt2, "quality_distribution.png")

    def error_rate_distribution(
        self,
        cer_values: list[float],
        wer_values: list[float],
    ) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2

        fig, axes = plt2.subplots(1, 2, figsize=(11, 4))
        for ax, vals, label, color in [
            (axes[0], cer_values, "CER", "tomato"),
            (axes[1], wer_values, "WER", "darkorange"),
        ]:
            ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.85)
            ax.set_title(f"{label} Distribution")
            ax.set_xlabel(label)
            ax.set_ylabel("Document Count")
            ax.grid(True, alpha=0.3)
        plt2.tight_layout()
        return self._save(plt2, "error_rate_distribution.png")

    # ── box plots by condition ────────────────────────────────────────────

    def quality_by_condition(
        self,
        quality_scores: list[float],
        condition_labels: list[str],
    ) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2

        unique = sorted(set(condition_labels))
        groups = [
            [q for q, c in zip(quality_scores, condition_labels) if c == cond]
            for cond in unique
        ]
        fig, ax = plt2.subplots(figsize=(max(8, len(unique) * 1.2), 5))
        ax.boxplot(groups, tick_labels=unique, patch_artist=True)
        ax.set_title("Quality Score Distribution by Condition")
        ax.set_xlabel("Quality Condition")
        ax.set_ylabel("OCR Quality Index")
        ax.axhline(60, color="red", linestyle="--", lw=1, label="Threshold")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt2.tight_layout()
        return self._save(plt2, "quality_by_condition.png")

    def wer_by_condition(
        self,
        wer_values: list[float],
        condition_labels: list[str],
    ) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2

        unique = sorted(set(condition_labels))
        groups = [
            [w for w, c in zip(wer_values, condition_labels) if c == cond]
            for cond in unique
        ]
        fig, ax = plt2.subplots(figsize=(max(8, len(unique) * 1.2), 5))
        ax.boxplot(groups, tick_labels=unique, patch_artist=True)
        ax.set_title("WER Distribution by Quality Condition")
        ax.set_xlabel("Quality Condition")
        ax.set_ylabel("Word Error Rate (WER)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
        plt2.tight_layout()
        return self._save(plt2, "wer_by_condition.png")

    # ── correlation heatmap ───────────────────────────────────────────────

    def correlation_heatmap(self, data: dict[str, list[float]]) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2
        import numpy as np

        keys = list(data.keys())
        n = len(keys)
        matrix = [[0.0] * n for _ in range(n)]

        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                xi, xj = data[ki], data[kj]
                mu_i, mu_j = sum(xi) / len(xi), sum(xj) / len(xj)
                num = sum((a - mu_i) * (b - mu_j) for a, b in zip(xi, xj))
                di = sum((a - mu_i) ** 2 for a in xi) ** 0.5
                dj = sum((b - mu_j) ** 2 for b in xj) ** 0.5
                matrix[i][j] = num / (di * dj) if di and dj else 0.0

        arr = np.array(matrix)
        fig, ax = plt2.subplots(figsize=(max(5, n), max(4, n - 1)))
        im = ax.imshow(arr, cmap="RdYlGn", vmin=-1, vmax=1)
        plt2.colorbar(im, ax=ax, label="Pearson r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(keys, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Feature Correlation Matrix")
        plt2.tight_layout()
        return self._save(plt2, "correlation_heatmap.png")

    # ── degradation sensitivity ───────────────────────────────────────────

    def degradation_sensitivity(
        self,
        conditions: list[str],
        mean_quality: list[float],
        mean_wer: list[float],
    ) -> Path:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt2
        import numpy as np

        x = np.arange(len(conditions))
        width = 0.4
        fig, ax1 = plt2.subplots(figsize=(max(8, len(conditions) * 1.2), 5))
        ax2 = ax1.twinx()

        ax1.bar(
            x - width / 2,
            mean_quality,
            width,
            label="Mean Quality",
            color="steelblue",
            alpha=0.8,
        )
        ax2.bar(
            x + width / 2,
            mean_wer,
            width,
            label="Mean WER",
            color="tomato",
            alpha=0.8,
        )

        ax1.set_xlabel("Quality Condition")
        ax1.set_ylabel("Mean Quality Score", color="steelblue")
        ax2.set_ylabel("Mean WER", color="tomato")
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=30, ha="right")
        ax1.set_title("Degradation Sensitivity Analysis")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.grid(True, alpha=0.3, axis="y")
        plt2.tight_layout()
        return self._save(plt2, "degradation_sensitivity.png")
