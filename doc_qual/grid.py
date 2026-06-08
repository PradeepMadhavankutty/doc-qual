"""Per-region / grid-based document quality scoring.

A page-level composite score masks localised quality problems — a single
blurry corner or a fold across a table cell can fail OCR for that region
while the rest of the page scores well.  This module divides the image
into an N×M grid and scores each cell independently, returning a heatmap
and a ``worst_cell`` alert.

Usage::

    from doc_qual.grid import score_image_grid

    result = score_image_grid("scan.jpg", rows=4, cols=4)
    print(result.worst_cell.score)        # lowest cell score
    print(result.worst_region_alert)      # True if below alert_threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from doc_qual.scorer import compute_doc_qual_score
from doc_qual.utils import ImageInput, load_grayscale_image

_MIN_CELL_PX = 32  # cells smaller than this (in either dimension) are skipped


@dataclass(frozen=True)
class GridCell:
    """Score and metadata for one grid cell."""

    row: int
    col: int
    score: float
    feature_scores: dict[str, float] = field(default_factory=dict)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) pixels

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "score": round(self.score, 2),
            "feature_scores": {k: round(v, 2) for k, v in self.feature_scores.items()},
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "w": self.bbox[2],
                "h": self.bbox[3],
            },
        }


@dataclass
class GridResult:
    """Result of grid-based quality scoring for a document image."""

    grid_rows: int
    grid_cols: int
    cells: list[GridCell]
    heatmap: np.ndarray  # shape (grid_rows, grid_cols), dtype float32
    page_score: float  # median of all valid cell scores
    worst_cell: GridCell | None
    worst_region_alert: bool
    alert_threshold: float
    skipped_cells: int  # cells too small to score

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "page_score": round(self.page_score, 2),
            "worst_region_alert": self.worst_region_alert,
            "alert_threshold": self.alert_threshold,
            "skipped_cells": self.skipped_cells,
            "worst_cell": self.worst_cell.to_dict() if self.worst_cell else None,
            "heatmap": [
                [round(float(v), 2) for v in row] for row in self.heatmap.tolist()
            ],
            "cells": [c.to_dict() for c in self.cells],
        }

    def ascii_heatmap(self) -> str:
        """Return a printable ASCII representation of the score heatmap."""
        shade = " ░▒▓█"
        lines = []
        for r in range(self.grid_rows):
            row_str = ""
            for c in range(self.grid_cols):
                v = float(self.heatmap[r, c])
                if v < 0:
                    row_str += "?"
                else:
                    idx = int(v / 100.0 * (len(shade) - 1))
                    row_str += shade[min(idx, len(shade) - 1)]
            lines.append(row_str)
        return "\n".join(lines)


def score_image_grid(
    image: ImageInput,
    *,
    rows: int = 4,
    cols: int = 4,
    weights: dict[str, float] | None = None,
    engine: str | None = None,
    alert_threshold: float = 40.0,
) -> GridResult:
    """Score a document image on a per-cell grid.

    Divides the image into ``rows × cols`` non-overlapping cells and runs
    the full feature pipeline on each cell.  Cells smaller than
    ``_MIN_CELL_PX`` in either dimension are skipped.

    Args:
        image: Image path (str/Path) or grayscale/colour numpy array.
        rows: Number of grid rows.
        cols: Number of grid columns.
        weights: Optional custom feature weights (passed to scorer).
        engine: Engine profile name for weight selection.
        alert_threshold: Score below which ``worst_region_alert`` fires.

    Returns:
        :class:`GridResult` containing per-cell scores, a heatmap array,
        and a ``worst_region_alert`` flag.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be ≥ 1, got rows={rows}, cols={cols}")

    gray = load_grayscale_image(image)
    h, w = gray.shape

    cell_h = h // rows
    cell_w = w // cols

    heatmap = np.full((rows, cols), fill_value=-1.0, dtype=np.float32)
    cells: list[GridCell] = []
    skipped = 0

    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            y1 = y0 + cell_h if r < rows - 1 else h
            x0 = c * cell_w
            x1 = x0 + cell_w if c < cols - 1 else w

            ch, cw = y1 - y0, x1 - x0
            if ch < _MIN_CELL_PX or cw < _MIN_CELL_PX:
                skipped += 1
                continue

            cell_img = gray[y0:y1, x0:x1]
            try:
                cell_result = compute_doc_qual_score(
                    cell_img,
                    threshold=alert_threshold,
                    weights=weights,
                    engine=engine,
                    verbose=False,
                )
                score = cell_result.ocr_score
                feat = dict(cell_result.feature_scores)
            except Exception:  # noqa: BLE001
                score = 0.0
                feat = {}

            heatmap[r, c] = float(score)
            cells.append(
                GridCell(
                    row=r,
                    col=c,
                    score=score,
                    feature_scores=feat,
                    bbox=(x0, y0, cw, ch),
                )
            )

    valid_scores = [c.score for c in cells]
    if valid_scores:
        page_score = float(np.median(valid_scores))
        worst_cell = min(cells, key=lambda c: c.score)
    else:
        page_score = 0.0
        worst_cell = None

    return GridResult(
        grid_rows=rows,
        grid_cols=cols,
        cells=cells,
        heatmap=heatmap,
        page_score=page_score,
        worst_cell=worst_cell,
        worst_region_alert=(
            worst_cell is not None and worst_cell.score < alert_threshold
        ),
        alert_threshold=alert_threshold,
        skipped_cells=skipped,
    )
