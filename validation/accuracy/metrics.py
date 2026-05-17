"""Ground-truth accuracy metrics: CER, WER, Levenshtein, exact match."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AccuracyResult:
    document_id: str
    cer: float
    wer: float
    exact_match: bool
    levenshtein_distance: int
    normalized_edit_distance: float
    reference_chars: int
    reference_words: int
    hypothesis_chars: int
    hypothesis_words: int

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "exact_match": self.exact_match,
            "levenshtein_distance": self.levenshtein_distance,
            "normalized_edit_distance": round(self.normalized_edit_distance, 6),
            "reference_chars": self.reference_chars,
            "reference_words": self.reference_words,
            "hypothesis_chars": self.hypothesis_chars,
            "hypothesis_words": self.hypothesis_words,
            "char_accuracy": round(max(0.0, 1.0 - self.cer), 4),
            "word_accuracy": round(max(0.0, 1.0 - self.wer), 4),
        }


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """Wagner-Fischer edit distance between two token sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for fair comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def compute_accuracy(
    document_id: str,
    reference: str,
    hypothesis: str,
) -> AccuracyResult:
    """Compute CER, WER, and exact-match between reference and hypothesis texts.

    Both inputs are normalised (lowercased, whitespace collapsed) before
    comparison so superficial formatting differences don't penalise scores.
    """
    ref_norm = _normalise(reference)
    hyp_norm = _normalise(hypothesis)

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    char_dist = _levenshtein(ref_chars, hyp_chars)
    word_dist = _levenshtein(ref_words, hyp_words)

    n_ref_chars = max(len(ref_chars), 1)
    n_ref_words = max(len(ref_words), 1)

    cer = char_dist / n_ref_chars
    wer = word_dist / n_ref_words
    ned = char_dist / max(len(ref_chars), len(hyp_chars), 1)

    return AccuracyResult(
        document_id=document_id,
        cer=min(cer, 1.0),
        wer=min(wer, 1.0),
        exact_match=(ref_norm == hyp_norm),
        levenshtein_distance=char_dist,
        normalized_edit_distance=ned,
        reference_chars=len(ref_chars),
        reference_words=len(ref_words),
        hypothesis_chars=len(hyp_chars),
        hypothesis_words=len(hyp_words),
    )
