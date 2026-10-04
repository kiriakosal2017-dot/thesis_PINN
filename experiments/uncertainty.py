"""Predictive intervals from the PI-NODE deep ensemble, with and without recalibration.

    python -m experiments.uncertainty

The ensemble is the set of independently seeded PI-NODE runs. Recalibration is fitted on the
source validation blocks and scored on the later test period, which is how it would be used in
service. A random split of the test period is reported alongside for comparison only: with
one-minute sampling the residuals are strongly autocorrelated, so a random split overstates
how well the exchangeability assumption behind conformal prediction holds.
"""
import json
from pathlib import Path

import numpy as np

from experiments.summarise import seed_runs
from experiments.train import load_everything
from pinode.training import metrics

LEVELS = (0.90, 0.95)
Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}


def temperature(resid, sigma):
    return float(np.sqrt(np.mean((resid / sigma) ** 2)))


def conformal_quantile(resid, sigma, level):
    scores = np.sort(np.abs(resid) / sigma)
    k = int(np.ceil((len(scores) + 1) * level))
    return float("inf") if k > len(scores) else float(scores[k - 1])


def coverage_width(mu, radius, y):
    return float(np.mean(np.abs(y - mu) <= radius) * 100.0), float(np.mean(2.0 * radius))


def evaluate(mu_c, sig_c, y_c, mu_e, sig_e, y_e):
    rows = {}
    T = temperature(y_c - mu_c, sig_c)
    for level in LEVELS:
        q = conformal_quantile(y_c - mu_c, sig_c, level)
        for method, radius in (("raw", Z[level] * sig_e), ("temperature", Z[level] * T * sig_e), ("conformal", q * sig_e)):
            rows[f"{method}@{level}"] = coverage_width(mu_e, radius, y_e)
    rows["T"] = T
    return rows


def main():
    source, scaler, train, val, test, targets = load_everything()
    runs = seed_runs("pinode")
    stack = {k: np.stack([r[k] for r in runs]) for k in ("val", "test")}
    mu = {k: v.mean(0) for k, v in stack.items()}
    sigma = {k: np.maximum(v.std(0, ddof=1), 1e-3) for k, v in stack.items()}
    y = {"val": val.y[val.ends].astype(float), "test": test.y[test.ends].astype(float)}

    out = {"members": len(runs),
           "ensemble_test": list(metrics(mu["test"], y["test"])),
           "mean_sigma_test": float(sigma["test"].mean()),
           "residual_autocorr_lag1": float(np.corrcoef((y["test"] - mu["test"])[:-1], (y["test"] - mu["test"])[1:])[0, 1])}
    out["chronological"] = evaluate(mu["val"], sigma["val"], y["val"], mu["test"], sigma["test"], y["test"])

    perm = np.random.default_rng(42).permutation(len(y["test"]))
    c, e = perm[: len(perm) // 2], perm[len(perm) // 2:]
    out["random_split"] = evaluate(mu["test"][c], sigma["test"][c], y["test"][c], mu["test"][e], sigma["test"][e], y["test"][e])

    Path("results_v2/uncertainty.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
