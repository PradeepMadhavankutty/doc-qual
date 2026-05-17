"""Dataset manifest — per-document metadata registry."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DocumentRecord:
    document_id: str
    image_path: str
    document_type: str
    quality_condition: str
    difficulty_level: str
    source_dataset: str
    ground_truth: str
    ground_truth_words: list[str]
    ground_truth_available: bool = True

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["word_count"] = len(self.ground_truth_words)
        d["char_count"] = len(self.ground_truth)
        return d


class DatasetManifest:
    """Ordered collection of DocumentRecord entries with JSON persistence."""

    def __init__(self) -> None:
        self._records: list[DocumentRecord] = []

    def add(self, record: DocumentRecord) -> None:
        self._records.append(record)

    def __iter__(self) -> Iterator[DocumentRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in self._records]
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        manifest = cls()
        for item in json.loads(path.read_text(encoding="utf-8")):
            item.pop("word_count", None)
            item.pop("char_count", None)
            manifest.add(DocumentRecord(**item))
        return manifest

    def summary(self) -> dict[str, object]:
        types: dict[str, int] = {}
        conditions: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        for r in self._records:
            types[r.document_type] = types.get(r.document_type, 0) + 1
            conditions[r.quality_condition] = conditions.get(r.quality_condition, 0) + 1
            difficulties[r.difficulty_level] = (
                difficulties.get(r.difficulty_level, 0) + 1
            )
        return {
            "total_documents": len(self._records),
            "document_types": types,
            "quality_conditions": conditions,
            "difficulty_levels": difficulties,
        }
