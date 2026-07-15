# Doc-Qual

`Doc-Qual` is a lightweight Python package for estimating whether a document image is ready for OCR before you spend time and money running OCR.

It computes a **0–100 OCR readiness score** from 11 interpretable image features. The score is **positively oriented**: a higher score always means better input quality and is expected to correlate with higher OCR extraction accuracy.

## Installation

```bash
pip install doc-qual
```

Optional extras:

```bash
pip install "doc-qual[calibration]"      # adds scipy for NNLS weight fitting
pip install "doc-qual[pdf]"              # adds pypdfium2 for PDF scoring
pip install "doc-qual[calibration,pdf]"  # everything
```

## Python Usage

```python
from doc_qual import compute_doc_qual_score

result = compute_doc_qual_score("scan.jpg", verbose=False)

print(result.ocr_score)         # 0–100
print(result.passed)            # True if score >= threshold (default 60)
print(result.feature_scores)   # per-feature breakdown
print(result.recommendations)  # actionable fix suggestions
```

### Engine-specific profiles

```python
result = compute_doc_qual_score("scan.jpg", engine="tesseract")
# also: "textract", "azure", "paddleocr", or path to custom JSON
```

### Per-region grid scoring

```python
from doc_qual import score_image_grid

grid = score_image_grid("scan.jpg", rows=4, cols=4, alert_threshold=40.0)
print(grid.page_score)          # median cell score
print(grid.worst_cell)          # GridCell with lowest score
print(grid.ascii_heatmap())     # visual quality map
```

### PDF scoring

```python
from doc_qual import score_pdf

pdf_result = score_pdf("document.pdf")
print(pdf_result.summary_score)   # median across pages
print(pdf_result.worst_page)      # page number with lowest score
```

## CLI Usage

```bash
# Score a single image
doc-qual scan.jpg

# Score with engine profile
doc-qual scan.jpg --engine tesseract

# Score a PDF
doc-qual document.pdf --dpi 200

# Per-region grid
doc-qual scan.jpg --grid 4x4 --alert-threshold 40

# JSON output
doc-qual scan.jpg --format json

# Exit code 1 if below threshold (useful in CI pipelines)
doc-qual scan.jpg --threshold 60

# Log ground-truth CER for calibration
doc-qual calibrate scan.jpg --ground-truth gt.json --engine tesseract

# Fit optimised weights from logged calibration data
doc-qual fit-weights --output-profile my_weights.json
```

## Score Orientation

Every feature score and the final composite score follow the same positive convention:

| Score key | What it measures | 0 means | 100 means | Inversion applied |
|---|---|---|---|---|
| `sharpness` | Laplacian variance (blur detection) | Extremely blurry | Razor-sharp | No — higher variance = sharper |
| `noise` | Gaussian residual std | Extremely noisy | Clean | `invert=True` |
| `skew` | Abs rotation angle (degrees) | Severely tilted | Perfectly straight | `invert=True` |
| `brightness` | Exposure + contrast balance | Too dark or blown-out | Ideal exposure | `100 − penalty` |
| `ridges` | Hessian eigenvalue (stroke structure) | Blank page | Rich text strokes | No — higher response = more text |
| `edges` | Canny edge density | Empty or over-saturated | Normal text density | Piecewise bell curve |
| `ink_bleedthrough` | Dark pixels in page margins | Severe bleed-through | Clean margins | `invert=True` |
| `shadow_gradient` | Cell brightness std across page | Severe lighting gradient | Perfectly uniform | `invert=True` |
| `local_contrast` | Michelson contrast in 64×64 blocks | Washed-out / faded ink | Sharp ink-paper separation | No — higher contrast = better |
| `crinkle_fold` | LoG energy in paper background | Severely folded / crinkled | Flat, clean | `invert=True` |
| `brisque` | MSCN coefficient deviation from Gaussian | Heavily distorted | Near-pristine | `100 − deviation` |

The composite `ocr_score` is a weighted average of the above — it is guaranteed to lie in `[0, 100]` and inherits the same positive orientation.

**Validated relationship:**

```
Higher data quality score  →  Better expected OCR extraction accuracy
Lower data quality score   →  More likely to degrade OCR output
```

Monotonicity is enforced by 28 automated tests in `tests/test_positive_orientation.py` that verify each metric degrades as the corresponding quality dimension worsens (blur, noise, skew, darkness, shadow, bleed-through, low contrast, crinkles).

## Feature Weights (defaults)

| Feature | Default weight | Notes |
|---|---|---|
| `sharpness` | 0.22 | Largest driver — blur is the #1 OCR killer |
| `noise` | 0.16 | |
| `edges` | 0.15 | |
| `skew` | 0.13 | |
| `brightness` | 0.13 | |
| `ridges` | 0.08 | |
| `ink_bleedthrough` | 0.04 | |
| `shadow_gradient` | 0.04 | |
| `local_contrast` | 0.03 | |
| `crinkle_fold` | 0.02 | |
| `brisque` | 0.00 | Off by default; enable via `weights=` or calibration |

Engine profiles (e.g. `engine="tesseract"`) shift these weights to reflect how each OCR backend responds to different quality defects.

## Development

```bash
git clone https://github.com/PradeepMadhavankutty/doc-qual.git
cd doc-qual
pip install -e ".[dev,calibration]"

pytest            # 161 tests
ruff check .
black --check .
python -m build
```

## License

MIT
