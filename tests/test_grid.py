"""Tests for per-region grid scoring."""

from __future__ import annotations

import numpy as np
import pytest

from doc_qual.grid import GridResult, score_image_grid


def test_grid_result_cell_count(clean_doc) -> None:  # type: ignore[no-untyped-def]
    result = score_image_grid(clean_doc, rows=3, cols=3)
    assert isinstance(result, GridResult)
    # All cells should be valid for a 300×500 image with 3×3 grid
    assert len(result.cells) == 9


def test_grid_heatmap_shape(clean_doc) -> None:  # type: ignore[no-untyped-def]
    result = score_image_grid(clean_doc, rows=4, cols=4)
    assert result.heatmap.shape == (4, 4)


def test_grid_scores_in_range(clean_doc) -> None:  # type: ignore[no-untyped-def]
    result = score_image_grid(clean_doc, rows=3, cols=3)
    for cell in result.cells:
        assert 0.0 <= cell.score <= 100.0


def test_grid_page_score_is_median(clean_doc) -> None:  # type: ignore[no-untyped-def]
    result = score_image_grid(clean_doc, rows=3, cols=3)
    scores = [c.score for c in result.cells]
    expected_median = float(np.median(scores))
    assert result.page_score == pytest.approx(expected_median, abs=0.1)


def test_grid_worst_cell_is_lowest(clean_doc) -> None:  # type: ignore[no-untyped-def]
    result = score_image_grid(clean_doc, rows=3, cols=3)
    assert result.worst_cell is not None
    min_score = min(c.score for c in result.cells)
    assert result.worst_cell.score == pytest.approx(min_score, abs=0.01)


def test_grid_alert_fires_when_worst_below_threshold() -> None:
    # Create a large-enough image where one quadrant is uniformly dark
    img = np.ones((200, 200), dtype=np.uint8) * 230
    img[:100, :100] = 5  # top-left quadrant → should get very low score
    result = score_image_grid(img, rows=2, cols=2, alert_threshold=50.0)
    assert result.worst_region_alert is True


def test_grid_alert_off_for_clean_doc(clean_doc) -> None:  # type: ignore[no-untyped-def]
    # Use alert_threshold=0 so clean doc never triggers the alert
    result = score_image_grid(clean_doc, rows=2, cols=2, alert_threshold=0.0)
    assert result.worst_region_alert is False


def test_grid_to_dict_serialisable(clean_doc) -> None:  # type: ignore[no-untyped-def]
    import json

    result = score_image_grid(clean_doc, rows=2, cols=2)
    d = result.to_dict()
    json.dumps(d)  # must not raise


def test_grid_ascii_heatmap_shape(clean_doc) -> None:  # type: ignore[no-untyped-def]
    result = score_image_grid(clean_doc, rows=3, cols=4)
    lines = result.ascii_heatmap().splitlines()
    assert len(lines) == 3
    assert all(len(line) == 4 for line in lines)


def test_grid_invalid_rows_raises() -> None:
    img = np.ones((100, 100), dtype=np.uint8) * 200
    with pytest.raises(ValueError, match="rows and cols"):
        score_image_grid(img, rows=0, cols=2)
