"""Tests for the accuracy metrics module."""

from __future__ import annotations

from validation.accuracy.metrics import compute_accuracy


def test_perfect_match() -> None:
    acc = compute_accuracy("doc_001", "Hello world", "Hello world")
    assert acc.cer == 0.0
    assert acc.wer == 0.0
    assert acc.exact_match is True


def test_case_insensitive_normalisation() -> None:
    acc = compute_accuracy("doc_002", "Hello World", "hello world")
    assert acc.exact_match is True


def test_whitespace_normalisation() -> None:
    acc = compute_accuracy("doc_003", "Hello  World", "Hello World")
    assert acc.exact_match is True


def test_single_char_deletion_cer() -> None:
    acc = compute_accuracy("doc_004", "Hello", "Hell")
    assert acc.cer == 1 / 5  # 1 char deleted out of 5


def test_single_word_deletion_wer() -> None:
    acc = compute_accuracy("doc_005", "Hello World", "Hello")
    assert acc.wer == 1 / 2  # 1 word deleted out of 2


def test_completely_wrong() -> None:
    acc = compute_accuracy("doc_006", "cat", "xyz")
    assert acc.exact_match is False
    assert acc.cer > 0.0
    assert acc.wer > 0.0


def test_empty_hypothesis() -> None:
    acc = compute_accuracy("doc_007", "Hello world", "")
    assert acc.cer == 1.0 or acc.cer > 0.9
    assert not acc.exact_match


def test_result_dict_has_required_keys() -> None:
    acc = compute_accuracy("doc_008", "Test text", "Test text")
    d = acc.to_dict()
    required = (
        "cer",
        "wer",
        "exact_match",
        "levenshtein_distance",
        "char_accuracy",
        "word_accuracy",
    )
    for key in required:
        assert key in d
