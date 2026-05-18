# Doc-Qual

`Doc-Qual` is a lightweight Python package for estimating whether a document image is ready for OCR before you spend time and money running OCR.

It computes a 0-100 OCR readiness score from interpretable image features:

| Feature | What it measures |
| --- | --- |
| Sharpness | Laplacian variance for blur detection |
| Noise | Gaussian residual noise estimate |
| Edges | Canny edge density for text-like structure |
| Skew | Hough-line based rotation estimate |
| Brightness | Exposure and contrast balance |
| Ridges | Hessian response for stroke-like structures |

## Installation

```bash
pip install doc-qual
```

For development:

```bash
git clone https://github.com/PradeepMadhavankutty/doc-qual.git
cd doc-qual
pip install -e ".[dev]"
```

## Python Usage

```python
from doc_qual import compute_doc_qual_score

result = compute_doc_qual_score("scan.jpg", verbose=False)

print(result.ocr_score)
print(result.passed)
print(result.recommendations)
```

## CLI Usage

```bash
doc-qual path/to/image.jpg
doc-qual path/to/image.jpg --format json
doc-qual path/to/image.jpg --threshold 60
```

When `--threshold` is provided, the CLI exits with code `1` if the image score is below the threshold. This makes it useful in CI and batch document-processing pipelines.

## Current Status

This is an alpha implementation with expert-calibrated default weights. The long-term research direction is to calibrate feature weights empirically against OCR character error rate across datasets and engines.

## Development

```bash
pytest
ruff check .
python -m build
```

## License

MIT
