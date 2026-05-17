"""Tests for the hypothesis testing module."""

from __future__ import annotations

import pytest

from validation.stats.hypothesis import validate_quality_accuracy_hypothesis


def _perfect_negative() -> tuple[list[float], list[float]]:
    """Quality up → error down: strong negative correlation."""
    qs = [float(i) for i in range(10, 100, 10)]
    ers = [float(1.0 - i / 100.0) for i in range(10, 100, 10)]
    return qs, ers


def _zero_correlation() -> tuple[list[float], list[float]]:
    qs = [50.0] * 10
    ers = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.15]
    return qs, ers


def test_strong_negative_correlation_rejects_h0() -> None:
    qs, ers = _perfect_negative()
    result = validate_quality_accuracy_hypothesis(qs, ers)
    assert result.pearson_r < -0.9
    assert result.reject_h0 is True


def test_zero_variance_quality_does_not_reject_h0() -> None:
    qs, ers = _zero_correlation()
    result = validate_quality_accuracy_hypothesis(qs, ers)
    assert result.reject_h0 is False


def test_result_dict_keys() -> None:
    qs, ers = _perfect_negative()
    result = validate_quality_accuracy_hypothesis(qs, ers)
    d = result.to_dict()
    for key in (
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "regression_r2",
        "reject_h0",
        "ci_lower",
        "ci_upper",
        "n",
    ):
        assert key in d


def test_spearman_agrees_with_pearson_on_monotone_data() -> None:
    qs, ers = _perfect_negative()
    result = validate_quality_accuracy_hypothesis(qs, ers)
    assert result.spearman_rho < -0.9


def test_requires_minimum_samples() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        validate_quality_accuracy_hypothesis([1.0, 2.0], [0.5, 0.3])


def test_condition_labels_run_anova() -> None:
    qs, ers = _perfect_negative()
    labels = ["clean", "blur"] * 5
    result = validate_quality_accuracy_hypothesis(qs, ers, condition_labels=labels)
    assert result.anova_f >= 0.0
