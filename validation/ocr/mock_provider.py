"""Mock OCR provider.

Simulates OCR extraction using ground-truth text with condition-calibrated
word-error rates.  Works without any OCR binary so the framework runs
everywhere out of the box; use it for CI and development.
"""

from __future__ import annotations

import random
from pathlib import Path

from validation.config import CONDITION_PROFILE
from validation.ocr.base import OCRProvider, OCRResult

_SUBSTITUTIONS = [
    ("a", "o"),
    ("e", "c"),
    ("l", "1"),
    ("0", "O"),
    ("i", "l"),
    ("rn", "m"),
    ("cl", "d"),
    ("S", "5"),
    ("B", "8"),
    ("Z", "2"),
]

_DELETIONS = ["the ", "and ", "of ", "to "]


def _corrupt_word(word: str, rng: random.Random) -> str:
    """Apply a single character-level substitution to simulate OCR noise."""
    if len(word) <= 1:
        return word
    choice = rng.random()
    if choice < 0.4:
        # character swap
        i = rng.randint(0, len(word) - 1)
        chars = list(word)
        for src, dst in _SUBSTITUTIONS:
            if src in word:
                return word.replace(src, dst, 1)
        chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)
    elif choice < 0.7:
        # character drop
        i = rng.randint(0, len(word) - 1)
        return word[:i] + word[i + 1 :]
    else:
        # character duplicate
        i = rng.randint(0, len(word) - 1)
        return word[:i] + word[i] + word[i:]


class MockOCRProvider(OCRProvider):
    """Deterministic mock OCR using ground-truth text + injected errors.

    The ground-truth lookup is injected at runtime by the pipeline so that
    the provider doesn't need database access at construction time.
    """

    name = "mock"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._ground_truth_map: dict[str, str] = {}
        self._condition_map: dict[str, str] = {}

    def register_document(
        self, document_id: str, ground_truth: str, quality_condition: str
    ) -> None:
        self._ground_truth_map[document_id] = ground_truth
        self._condition_map[document_id] = quality_condition

    def _extract(self, image_path: Path, document_id: str) -> OCRResult:
        ground_truth = self._ground_truth_map.get(document_id, "")
        condition = self._condition_map.get(document_id, "clean")
        _, wer_rate = CONDITION_PROFILE.get(condition, ("easy", 0.05))  # type: ignore[call-overload]

        rng = random.Random(self._seed + hash(document_id) % (2**31))
        words = ground_truth.split()
        corrupted: list[str] = []
        word_confs: list[float] = []

        for word in words:
            if rng.random() < wer_rate:
                corrupted.append(_corrupt_word(word, rng))
                conf = max(0.0, rng.gauss(0.55, 0.15))
            else:
                corrupted.append(word)
                conf = min(1.0, rng.gauss(0.94, 0.04))
            word_confs.append(conf)

        # Occasionally drop or duplicate a word
        if rng.random() < wer_rate * 0.5 and corrupted:
            i = rng.randint(0, len(corrupted) - 1)
            corrupted.pop(i)
            word_confs.pop(i) if i < len(word_confs) else None

        extracted = " ".join(corrupted)
        avg_conf = sum(word_confs) / max(len(word_confs), 1)

        return OCRResult(
            document_id=document_id,
            provider=self.name,
            extracted_text=extracted,
            confidence=avg_conf,
            word_confidences=word_confs,
            success=True,
        )
