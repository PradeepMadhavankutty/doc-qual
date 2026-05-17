"""Synthetic document dataset generation and manifest management."""

from validation.dataset.generator import DocumentGenerator
from validation.dataset.manifest import DatasetManifest, DocumentRecord

__all__ = ["DocumentGenerator", "DatasetManifest", "DocumentRecord"]
