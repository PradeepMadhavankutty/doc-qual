# Changelog

## 0.2.0

### New features
- **Engine-specific weight profiles** — pass `engine="tesseract"`, `"textract"`, `"azure"`, or `"paddleocr"` to `compute_doc_qual_score()` and the CLI to use empirically tuned feature weights for each OCR backend. Custom JSON profiles are also supported.
- **Calibration mode** — `doc-qual calibrate` logs per-image feature scores alongside ground-truth CER to `~/.doc_qual/calibration.csv`. `doc-qual fit-weights` runs NNLS regression on the logged data and writes an optimised JSON weight profile.
- **Per-region grid scoring** — `score_image_grid(image, rows=4, cols=4)` returns a `GridResult` with per-cell scores, an ASCII heat-map, worst-cell identification, and an optional alert when any region falls below a threshold.
- **PDF support** — `score_pdf(path)` renders each page via `pypdfium2` (pure-Python wheel) or `pdf2image`+poppler and returns a `PDFQualityResult` with per-page breakdown and a median summary score.
- **Five new feature extractors**:
  - `ink_bleedthrough` — detects ink bleeding through thin paper from the reverse side.
  - `shadow_gradient` — detects uneven lighting / shadow gradients using a coarse brightness grid.
  - `local_contrast` — Michelson contrast in 64×64 blocks; flags low-contrast scans.
  - `crinkle_fold` — Laplacian-of-Gaussian on paper background regions to detect fold lines and crinkles.
  - `brisque_like` — no-reference perceptual quality via MSCN coefficient statistics (pure NumPy, no opencv-contrib required).
- **Crop-aware blur and edge metrics** — blur and edge scores are computed on isolated text-region crops when detectable, falling back to full-image analysis with a logged warning.

### Improvements
- `OCRQualityResult` gains an `engine` field (serialised in `to_dict()`).
- `compute_doc_qual_score` normalises by the sum of *active* weights so profiles that omit features are handled correctly.
- CLI extended with `--engine`, `--grid ROWSxCOLS`, `--alert-threshold`, `--dpi`, and `--max-pages` flags; old single-argument call style (`doc-qual image.jpg`) remains fully backwards-compatible.

### Dependencies (optional extras)
- `pip install doc-qual[calibration]` — adds `scipy` for NNLS weight fitting.
- `pip install doc-qual[pdf]` — adds `pypdfium2` for PDF rendering.
- `pip install doc-qual[pdf-poppler]` — adds `pdf2image` as an alternative PDF renderer.

---

## 0.1.0

- Initial alpha release.
- Added composite Doc-Qual score.
- Added feature modules for sharpness, noise, edges, skew, brightness, and ridges.
- Added CLI entrypoint.
- Added tests, docs, and GitHub Actions workflows.
