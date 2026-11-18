"""Statistics for comparing Snake policies.

Why these methods?
- Win rate is a binomial proportion. The textbook interval p +/- z*sqrt(p(1-p)/n) collapses to width 0 at
  p = 0 or 1 and undercovers near the edges, exactly where Snake win rates live (0% for baselines, ~100% for
  good agents). The Wilson score interval keeps correct coverage there (Brown, Cai & DasGupta, 2001).
- Scores and step counts are skewed and bounded, so we use a percentile bootstrap for their means instead of
  assuming normality.
- Two policies' win rates are compared with a two-proportion z-test, and with Fisher's exact test when
  counts are small (the z-test's normal approximation is poor with < 5 expected wins or losses).
- Step counts of two policies are compared with the Mann-Whitney U test (no normality assumption).
"""
import math

import numpy as np
from scipy import stats as sps


def wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval for a binomial proportion -> (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    z = sps.norm.ppf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values, stat=np.mean, n_boot=10000, confidence=0.95, seed=0):
    """Percentile bootstrap CI of `stat` -> (estimate, low, high). NaNs for an empty sample."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    alpha = (1 - confidence) / 2
    return (float(stat(x)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha)))


def two_proportion_test(k1, n1, k2, n2):
    """Two-sided pooled two-proportion z-test -> (z, p_value). p = 1 when both proportions are 0 or 1."""
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / se
    return z, float(2 * sps.norm.sf(abs(z)))


def fisher_exact(k1, n1, k2, n2):
    """Two-sided Fisher exact test on the 2x2 win/loss table -> p_value."""
    return float(sps.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue)


def mann_whitney(x, y):
    """Two-sided Mann-Whitney U test -> (U, p_value); NaNs if either sample is empty."""
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")
    res = sps.mannwhitneyu(x, y, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)
