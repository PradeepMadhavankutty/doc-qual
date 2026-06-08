"""Text-region crop detection and crop-level blur / edge metrics.

This module isolates the geometry of finding text-like regions inside a
document image and computing per-crop blur and edge measurements.  The
public :func:`calculate_document_blur_edge_metrics` returns the averaged
document-level metrics consumed by :mod:`doc_qual.features.blurriness`
and :mod:`doc_qual.features.edges`.

Backward compatibility:
    When no valid text crops are detected (e.g. a blank image, or an
    image where thresholding fails), the helpers fall back to computing
    blur / edge on the full image so the package never silently returns
    zero scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── crop filtering rules ───────────────────────────────────────────────────
MIN_CROP_WIDTH: int = 20
MIN_CROP_HEIGHT: int = 10
MIN_CROP_AREA: int = 200
MAX_BLANK_PIXEL_RATIO: float = 0.95
BLANK_PIXEL_THRESHOLD: int = 240  # pixels brighter than this count as "blank"
OVERLAP_KEEP_RATIO: float = 0.70  # drop a box if a larger box covers >70 % of it


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit grayscale view of ``image`` without copying when possible."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expected a 2D grayscale or 3D color image")

    if gray.dtype != np.uint8:
        out = np.empty_like(gray, dtype=np.uint8)
        cv2.normalize(gray, out, 0, 255, cv2.NORM_MINMAX)
        gray = out
    return gray


def _deduplicate_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Drop boxes that are largely contained within a strictly larger box."""
    if not boxes:
        return boxes
    # Sort by area, largest first.  Keep larger boxes by default.
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted_boxes:
        x, y, w, h = box
        area = w * h
        absorbed = False
        for kx, ky, kw, kh in kept:
            ix1 = max(x, kx)
            iy1 = max(y, ky)
            ix2 = min(x + w, kx + kw)
            iy2 = min(y + h, ky + kh)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if area > 0 and inter / area > OVERLAP_KEEP_RATIO:
                absorbed = True
                break
        if not absorbed:
            kept.append(box)
    # Restore reading order (top-to-bottom, left-to-right).
    kept.sort(key=lambda b: (b[1], b[0]))
    return kept


def _is_valid_crop(crop: np.ndarray) -> bool:
    """Apply the documented filtering rules to a single crop."""
    h, w = crop.shape[:2]
    if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
        return False
    if w * h < MIN_CROP_AREA:
        return False
    if crop.size == 0:
        return False
    blank_ratio = float(np.count_nonzero(crop > BLANK_PIXEL_THRESHOLD)) / float(
        crop.size
    )
    if blank_ratio > MAX_BLANK_PIXEL_RATIO:
        return False
    return True


def detect_text_crops(image: np.ndarray) -> list[dict[str, Any]]:
    """Detect text-like regions in a document image.

    The pipeline is:
        grayscale → Otsu threshold (inverted so text is foreground) →
        morphological dilation → contour detection → bounding boxes →
        filtering and deduplication.

    Args:
        image: Grayscale or BGR document image as a numpy array.

    Returns:
        A list of dicts, each ``{"bbox": [x, y, w, h], "crop": ndarray}``,
        sorted in approximate reading order.  Returns an empty list when
        no qualifying region is found.
    """
    gray = _to_grayscale(image)
    height, width = gray.shape

    # Otsu finds a global threshold; INV makes dark text -> white foreground.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Wide-but-shallow kernel merges characters into word/line blobs without
    # bridging across separate text lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Reject anything that spans almost the entire page — it is either
        # the page border or a thresholding failure.
        if w >= int(width * 0.98) and h >= int(height * 0.98):
            continue
        raw_boxes.append((x, y, w, h))

    deduped = _deduplicate_boxes(raw_boxes)

    crops: list[dict[str, Any]] = []
    for x, y, w, h in deduped:
        crop = gray[y : y + h, x : x + w]
        if _is_valid_crop(crop):
            crops.append({"bbox": [int(x), int(y), int(w), int(h)], "crop": crop})
    return crops


def calculate_crop_blur_score(crop: np.ndarray) -> float:
    """Laplacian-variance blur score for a single crop.

    Higher values indicate a sharper crop.
    """
    gray = _to_grayscale(crop)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_crop_edge_score(crop: np.ndarray) -> float:
    """Canny edge-density score for a single crop in ``[0, 1]``.

    The density is computed as ``non-zero edge pixels / total pixels``.
    """
    gray = _to_grayscale(crop)
    edges = cv2.Canny(gray, 100, 200)
    if edges.size == 0:
        return 0.0
    return float(np.count_nonzero(edges)) / float(edges.size)


def _full_image_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _full_image_edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    if edges.size == 0:
        return 0.0
    return float(np.count_nonzero(edges)) / float(edges.size)


def calculate_document_blur_edge_metrics(
    image: np.ndarray,
    *,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Detect text crops and aggregate per-crop blur and edge metrics.

    Args:
        image: Grayscale or BGR document image.
        debug_dir: Optional directory.  When provided, each valid crop is
            written to disk as ``crop_<id>.png`` for inspection.  Crops are
            never persisted by default.

    Returns:
        A dict with the document-level averages, the number of detected
        and valid crops, and a per-crop record list.  When no valid crops
        survive filtering, the function falls back to full-image metrics
        and adds a ``warning`` key.
    """
    gray = _to_grayscale(image)
    crops = detect_text_crops(gray)

    crop_metrics: list[dict[str, Any]] = []
    blur_scores: list[float] = []
    edge_scores: list[float] = []

    debug_path: Path | None = None
    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    for idx, crop_info in enumerate(crops, start=1):
        crop = crop_info["crop"]
        blur = calculate_crop_blur_score(crop)
        edge = calculate_crop_edge_score(crop)
        blur_scores.append(blur)
        edge_scores.append(edge)
        crop_metrics.append(
            {
                "crop_id": idx,
                "bbox": crop_info["bbox"],
                "blur_score": round(blur, 4),
                "edge_score": round(edge, 6),
            }
        )
        if debug_path is not None:
            cv2.imwrite(str(debug_path / f"crop_{idx:03d}.png"), crop)

    result: dict[str, Any] = {
        "crop_count": len(crops),
        "valid_crop_count": len(crop_metrics),
        "crop_metrics": crop_metrics,
    }

    if blur_scores and edge_scores:
        result["document_blur_score"] = float(np.mean(blur_scores))
        result["document_edge_score"] = float(np.mean(edge_scores))
        result["fallback_used"] = False
    else:
        result["document_blur_score"] = _full_image_blur(gray)
        result["document_edge_score"] = _full_image_edge_density(gray)
        result["fallback_used"] = True
        result["warning"] = "No valid text crops detected. Full image metrics used."

    return result


__all__ = [
    "MIN_CROP_WIDTH",
    "MIN_CROP_HEIGHT",
    "MIN_CROP_AREA",
    "MAX_BLANK_PIXEL_RATIO",
    "BLANK_PIXEL_THRESHOLD",
    "calculate_crop_blur_score",
    "calculate_crop_edge_score",
    "calculate_document_blur_edge_metrics",
    "detect_text_crops",
]
