"""Post-hoc recalibration of PI-NODE predictive uncertainty (step 14).

Two methods, fit on a held-out calibration split and evaluated on a disjoint half:
  * temperature scaling — Gaussian variance matching: a single scalar T = sqrt(mean((y-mu)^2/sigma^2))
    rescales sigma; intervals are mu +/- z(level)*T*sigma. Parametric, assumes Gaussian residuals.
  * split conformal (normalized) — distribution-free: radius q*sigma where q is the finite-sample
    (1-alpha) quantile of the nonconformity scores |y-mu|/sigma. Guaranteed marginal coverage,
    adaptive width.

No retraining: operates on predictions from the existing ensemble / MC-Dropout (see main()).
"""
import numpy as np

# Two-sided Gaussian z for the target central-coverage levels (norm.ppf((1+level)/2)).
Z_TWO_SIDED = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}

# Floor on sigma (kW) before any division, to avoid blow-up at over-confident points.
SIGMA_FLOOR = 1e-6


def _safe_sigma(sigma):
    return np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)


def temperature_factor(resid, sigma):
    """NLL-optimal Gaussian variance-matching scalar: T = sqrt(mean((resid/sigma)^2))."""
    z = np.asarray(resid, dtype=float) / _safe_sigma(sigma)
    return float(np.sqrt(np.mean(z ** 2)))


def conformal_quantile(resid, sigma, alpha):
    """Finite-sample (1-alpha) quantile of normalized scores |resid|/sigma.

    Uses the split-conformal order statistic k = ceil((n+1)(1-alpha)); returns +inf if k>n
    (calibration set too small for the requested level).
    """
    s = np.abs(np.asarray(resid, dtype=float)) / _safe_sigma(sigma)
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(s)[k - 1])


def interval_coverage_width(mu, radius, true):
    """Empirical coverage and mean interval width for mu +/- radius (radius is per-point)."""
    mu = np.asarray(mu, dtype=float)
    radius = np.asarray(radius, dtype=float)
    true = np.asarray(true, dtype=float)
    within = np.abs(true - mu) <= radius
    return float(within.mean()), float((2.0 * radius).mean())
