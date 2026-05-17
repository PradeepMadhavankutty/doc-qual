"""Statistical hypothesis testing for OCR quality vs. accuracy.

H1: Higher OCR Quality Index leads to lower OCR extraction error rate.
H0: OCR Quality Index has no statistically significant relationship with
    OCR extraction accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class HypothesisResult:
    hypothesis: str
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    regression_slope: float
    regression_intercept: float
    regression_r2: float
    anova_f: float
    anova_p: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    reject_h0: bool
    interpretation: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "n": self.n,
            "pearson_r": round(self.pearson_r, 6),
            "pearson_p": round(self.pearson_p, 6),
            "spearman_rho": round(self.spearman_rho, 6),
            "spearman_p": round(self.spearman_p, 6),
            "regression_slope": round(self.regression_slope, 6),
            "regression_intercept": round(self.regression_intercept, 6),
            "regression_r2": round(self.regression_r2, 6),
            "anova_f": round(self.anova_f, 4),
            "anova_p": round(self.anova_p, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "confidence_level": self.confidence_level,
            "reject_h0": self.reject_h0,
            "interpretation": self.interpretation,
            "notes": self.notes,
        }


# ── pure-Python statistical primitives ───────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: list[float], ddof: int = 1) -> float:
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / max(len(xs) - ddof, 1)


def _std(xs: list[float], ddof: int = 1) -> float:
    return math.sqrt(_variance(xs, ddof))


def _cov(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(len(xs) - 1, 1)


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    sx, sy = _std(xs), _std(ys)
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = _cov(xs, ys) / (sx * sy)
    r = max(-1.0, min(1.0, r))
    n = len(xs)
    # t-distribution approximation for p-value
    if abs(r) >= 1.0 or n <= 2:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r**2))
    p = _t_pvalue(t, n - 2)
    return r, p


def _t_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value from Student-t distribution (Abramowitz & Stegun approx)."""
    x = df / (df + t**2)
    # Regularised incomplete beta function approximation
    try:
        from math import lgamma

        lbeta = lgamma(df / 2) + lgamma(0.5) - lgamma((df + 1) / 2)
        # Simple series approximation for small |t|
        p_one_tail = 0.5 * math.exp(
            math.log(max(x, 1e-300)) * (df / 2)
            + math.log(max(1 - x, 1e-300)) * 0.5
            - lbeta
        )
        return min(1.0, 2.0 * p_one_tail)
    except (ValueError, ZeroDivisionError):
        return 1.0


def _rank(xs: list[float]) -> list[float]:
    indexed = sorted(enumerate(xs), key=lambda kv: kv[1])
    ranks: list[float] = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    return _pearson(_rank(xs), _rank(ys))


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r²)."""
    cov = _cov(xs, ys)
    var_x = _variance(xs)
    if var_x == 0:
        return 0.0, _mean(ys), 0.0
    slope = cov / var_x
    intercept = _mean(ys) - slope * _mean(xs)
    # r²
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - _mean(ys)) ** 2 for y in ys)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    return slope, intercept, r2


def _one_way_anova(groups: list[list[float]]) -> tuple[float, float]:
    """Return (F-statistic, p-value) for a one-way ANOVA."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return 0.0, 1.0
    grand = [x for g in groups for x in g]
    grand_mean = _mean(grand)
    k = len(groups)
    n = len(grand)

    ss_between = sum(len(g) * (_mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum((x - _mean(g)) ** 2 for g in groups for x in g)
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or ss_within == 0:
        return 0.0, 1.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f = ms_between / max(ms_within, 1e-10)

    # Approximate p-value via Fisher's F distribution CDF (incomplete beta)
    x = df_between * f / (df_between * f + df_within)
    try:
        from math import lgamma

        lbeta = (
            lgamma(df_between / 2)
            + lgamma(df_within / 2)
            - lgamma((df_between + df_within) / 2)
        )
        p = math.exp(
            math.log(max(x, 1e-300)) * (df_between / 2)
            + math.log(max(1 - x, 1e-300)) * (df_within / 2)
            - lbeta
        )
        return f, min(1.0, p)
    except (ValueError, ZeroDivisionError):
        return f, 1.0


def _pearson_ci(r: float, n: int, alpha: float) -> tuple[float, float]:
    """Fisher z-transform confidence interval for Pearson r."""
    if n <= 3:
        return -1.0, 1.0
    z = 0.5 * math.log((1 + r) / max(1 - r, 1e-10))
    se = 1.0 / math.sqrt(n - 3)
    # z_crit for common alpha values (normal approximation)
    z_crit = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(alpha, 1.960)
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
    return r_lo, r_hi


# ── public API ─────────────────────────────────────────────────────────────


def validate_quality_accuracy_hypothesis(
    quality_scores: list[float],
    error_rates: list[float],
    confidence_level: float = 0.95,
    condition_labels: list[str] | None = None,
) -> HypothesisResult:
    """Test H1: quality ↑ → error rate ↓ (negative correlation expected).

    Args:
        quality_scores: OCR quality index per document (0–100).
        error_rates: CER or WER per document (0–1).
        confidence_level: e.g. 0.95 for 95 % CI.
        condition_labels: Optional quality-condition label per document for ANOVA.
    """
    n = len(quality_scores)
    if n < 4:
        raise ValueError(f"Need at least 4 data points, got {n}")

    r, p_pearson = _pearson(quality_scores, error_rates)
    rho, p_spearman = _spearman(quality_scores, error_rates)
    slope, intercept, r2 = _linear_regression(quality_scores, error_rates)
    ci_lo, ci_hi = _pearson_ci(r, n, confidence_level)
    alpha = 1.0 - confidence_level

    # One-way ANOVA across condition groups if labels provided
    if condition_labels:
        label_set = sorted(set(condition_labels))
        groups = [
            [e for e, lbl in zip(error_rates, condition_labels) if lbl == label]
            for label in label_set
        ]
        f_stat, p_anova = _one_way_anova(groups)
    else:
        f_stat, p_anova = _one_way_anova([[e] for e in error_rates])

    reject_h0 = p_pearson < alpha and r < -0.1

    # Build interpretation
    strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
    direction = "negative" if r < 0 else "positive"
    sig = "statistically significant" if p_pearson < alpha else "not significant"
    interpretation = (
        f"Pearson r={r:.3f} indicates a {strength} {direction} correlation "
        f"between quality score and error rate ({sig}, p={p_pearson:.4f}). "
        f"Linear regression: error_rate = {slope:.4f}·quality + {intercept:.4f} "
        f"(R²={r2:.3f})."
    )
    notes = [
        f"H0 {'rejected' if reject_h0 else 'not rejected'} at α={alpha:.2f}.",
        f"95% CI for Pearson r: [{ci_lo:.3f}, {ci_hi:.3f}].",
        f"Spearman ρ={rho:.3f} (p={p_spearman:.4f}).",
    ]

    return HypothesisResult(
        hypothesis="H1: Higher OCR Quality Index → lower OCR error rate",
        n=n,
        pearson_r=r,
        pearson_p=p_pearson,
        spearman_rho=rho,
        spearman_p=p_spearman,
        regression_slope=slope,
        regression_intercept=intercept,
        regression_r2=r2,
        anova_f=f_stat,
        anova_p=p_anova,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        confidence_level=confidence_level,
        reject_h0=reject_h0,
        interpretation=interpretation,
        notes=notes,
    )
