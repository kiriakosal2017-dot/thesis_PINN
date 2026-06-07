"""Unit checks for the post-hoc calibration math in calibrate_uncertainty."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from calibrate_uncertainty import (
    temperature_factor, conformal_quantile, interval_coverage_width,
    _safe_sigma, Z_TWO_SIDED, SIGMA_FLOOR,
)


def test_temperature_recovers_scalar_inflation():
    # If true spread is c*sigma, the NLL-optimal temperature should recover approximately c.
    rng = np.random.default_rng(0)
    n = 40000
    mu = rng.normal(0, 100, n)
    sigma = np.full(n, 50.0)
    c = 2.3
    true = mu + rng.normal(0, c * 50.0, n)
    T = temperature_factor(true - mu, sigma)
    assert abs(T - c) < 0.1, f"T={T}, expected ~{c}"


def test_conformal_achieves_nominal_coverage_when_raw_does_not():
    # Conformal calibration must bring an under-confident model up to the nominal coverage level.
    rng = np.random.default_rng(1)
    n = 40000
    mu = rng.normal(0, 100, n)
    sigma = np.full(n, 50.0)
    true = mu + rng.normal(0, 2.0 * 50.0, n)  # reported sigma is 2x too small
    resid = true - mu
    idx = rng.permutation(n)
    cal, ev = idx[: n // 2], idx[n // 2:]
    level = 0.90
    z = Z_TWO_SIDED[level]
    raw_cov, _ = interval_coverage_width(mu[ev], z * sigma[ev], true[ev])
    assert raw_cov < 0.80, f"raw coverage {raw_cov} should be clearly < nominal"
    q = conformal_quantile(resid[cal], sigma[cal], alpha=1 - level)
    conf_cov, conf_w = interval_coverage_width(mu[ev], q * sigma[ev], true[ev])
    assert abs(conf_cov - level) < 0.03, f"conformal coverage {conf_cov} not ~{level}"
    assert conf_w > 0


def test_conformal_quantile_finite_sample_correction():
    # With n=9 calibration points the finite-sample correction clips at the largest score
    # for alpha=0.10 and returns infinity when the required rank exceeds n.
    sigma = np.ones(9)
    resid = np.arange(1, 10, dtype=float)  # |resid|/sigma = 1..9
    # n=9, alpha=0.10 -> k=ceil(10*0.90)=9 <= 9 -> 9th (largest) score.
    assert conformal_quantile(resid, sigma, alpha=0.10) == 9.0
    # n=9, alpha=0.05 -> k=ceil(10*0.95)=10 > 9 -> not enough data -> inf.
    assert conformal_quantile(resid, sigma, alpha=0.05) == float("inf")


def test_safe_sigma_floors_small_values():
    # Near-zero sigma values cause division instability; they must be clipped to SIGMA_FLOOR.
    s = _safe_sigma(np.array([0.0, 1e-12, 5.0]))
    assert (s >= SIGMA_FLOOR).all()
    assert s[2] == 5.0


def test_z_two_sided_values():
    # Spot-check the z-table constants used for interval construction.
    assert abs(Z_TWO_SIDED[0.90] - 1.6448536) < 1e-4
    assert abs(Z_TWO_SIDED[0.95] - 1.9599640) < 1e-4


if __name__ == "__main__":
    test_temperature_recovers_scalar_inflation()
    test_conformal_achieves_nominal_coverage_when_raw_does_not()
    test_conformal_quantile_finite_sample_correction()
    test_safe_sigma_floors_small_values()
    test_z_two_sided_values()
    print("all calibration unit tests passed")
