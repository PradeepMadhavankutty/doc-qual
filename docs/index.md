# Doc-Qual

`Doc-Qual` scores document images for OCR readiness using fast, interpretable computer vision features.

## Install

```bash
pip install Doc-Qual
```

## Quick Start

```python
from doc_qual import compute_doc_qual_score

result = compute_doc_qual_score("scan.jpg", verbose=False)
print(result.to_dict())
```

## Command Line

```bash
doc-qual scan.jpg --format json --threshold 60
```
