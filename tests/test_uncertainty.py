"""Checks of the recalibration arithmetic used for the prediction intervals."""
import numpy as np

from experiments.uncertainty import Z, conformal_quantile, coverage_width, temperature


def test_temperature_recovers_scalar_inflation():
    rng = np.random.default_rng(0)
    n = 40000
    sigma = np.full(n, 50.0)
    resid = rng.normal(0, 2.3 * 50.0, n)
    assert abs(temperature(resid, sigma) - 2.3) < 0.1


def test_conformal_restores_nominal_coverage():
    rng = np.random.default_rng(1)
    n = 40000
    mu = rng.normal(0, 100, n)
    sigma = np.full(n, 50.0)
    y = mu + rng.normal(0, 2.0 * 50.0, n)          # the reported sigma is twice too small
    idx = rng.permutation(n)
    cal, ev = idx[: n // 2], idx[n // 2:]
    raw, _ = coverage_width(mu[ev], Z[0.90] * sigma[ev], y[ev])
    assert raw < 80.0
    q = conformal_quantile(y[cal] - mu[cal], sigma[cal], 0.90)
    cov, width = coverage_width(mu[ev], q * sigma[ev], y[ev])
    assert abs(cov - 90.0) < 3.0 and width > 0


def test_conformal_quantile_needs_enough_calibration_points():
    assert conformal_quantile(np.ones(5), np.ones(5), 0.95) == float("inf")
