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
import argparse
import csv
import glob
from pathlib import Path

from evaluate_uncertainty import build_model, load_into
from pinode_common import load_danae_temporal_sequences, make_loaders, predict_power, rmse_mape

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


LEVELS = (0.90, 0.95)
SPLIT_SEED = 42


def _ensemble_mu_sigma(proc, fi, calm, weather, input_size, test_loader, glob_pat):
    paths = sorted(glob.glob(glob_pat))
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 ensemble checkpoints matching '{glob_pat}', found {len(paths)}")
    runs, true = [], None
    for p in paths:
        m = build_model(proc, fi, calm, weather, input_size)
        m = load_into(m, p)
        preds, true = predict_power(m, test_loader)
        runs.append(preds)
    runs = np.stack(runs)  # (M, N)
    return runs.mean(0), runs.std(0), true, len(paths)


def _mcdropout_mu_sigma(model, test_loader, k):
    runs, true = [], None
    for _ in range(k):
        preds, true = predict_power(model, test_loader, mc_dropout=True)
        runs.append(preds)
    runs = np.stack(runs)  # (K, N)
    return runs.mean(0), runs.std(0), true


def _evaluate_estimator(tag, mu, sigma, true, rows):
    """Fit T and q on a random calibration half; report coverage/width on the eval half."""
    n = mu.size
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)
    cal, ev = perm[: n // 2], perm[n // 2:]
    resid_cal = true[cal] - mu[cal]
    sig_cal = sigma[cal]
    T = temperature_factor(resid_cal, sig_cal)  # level-independent
    rmse, mape = rmse_mape(mu, true)
    print(f"\n=== {tag}  (RMSE {rmse:.2f} kW, MAPE {mape:.2f}%; T={T:.3f}; n_cal={cal.size}, n_eval={ev.size}) ===")
    print(f"{'level':>6} {'method':>12} {'coverage':>9} {'mean_width_kW':>14}")
    for level in LEVELS:
        alpha = 1.0 - level
        z = Z_TWO_SIDED[level]
        q = conformal_quantile(resid_cal, sig_cal, alpha)
        radii = {
            "raw": z * sigma[ev],
            "temperature": z * T * sigma[ev],
            "conformal": q * sigma[ev],
        }
        for method, radius in radii.items():
            cov, width = interval_coverage_width(mu[ev], radius, true[ev])
            print(f"{level:>6.2f} {method:>12} {cov*100:>8.1f}% {width:>14.1f}")
            rows.append({"estimator": tag, "method": method, "target": level,
                         "empirical_coverage": round(cov, 4), "mean_width_kw": round(width, 1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_model_PI_NODE_danae.pt")
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--ensemble-glob", default="results/best_model_PI_NODE_seed*.pt")
    args = ap.parse_args()

    proc, fi, calm, weather, train_tuple, test_tuple = load_danae_temporal_sequences()
    input_size = train_tuple[0].shape[2]
    base = build_model(proc, fi, calm, weather, input_size)
    _, _, test_loader = make_loaders(base, train_tuple, test_tuple)

    print("=" * 70)
    print("PI-NODE Uncertainty Recalibration (DANAE test set, random 50/50 split)")
    print("=" * 70)

    rows = []

    # Deep ensemble (primary)
    mu_e, sig_e, true_e, m = _ensemble_mu_sigma(proc, fi, calm, weather, input_size,
                                                test_loader, args.ensemble_glob)
    _evaluate_estimator(f"Deep Ensemble (M={m})", mu_e, sig_e, true_e, rows)

    # MC-Dropout (secondary; same calibration machinery)
    model = load_into(base, args.checkpoint)
    mu_d, sig_d, true_d = _mcdropout_mu_sigma(model, test_loader, args.mc_samples)
    _evaluate_estimator(f"MC-Dropout (K={args.mc_samples})", mu_d, sig_d, true_d, rows)

    out = Path("results"); out.mkdir(exist_ok=True)
    csv_path = out / "calibration_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["estimator", "method", "target",
                                          "empirical_coverage", "mean_width_kw"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
