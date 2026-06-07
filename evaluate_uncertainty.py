"""Predictive uncertainty evaluation for the PI-NODE (source domain: DANAE).

Two complementary estimates of predictive uncertainty:

  1. MC-Dropout  : keep dropout active at inference, run K stochastic forward passes
                   on a single trained model. Cheap; captures model uncertainty.
  2. Deep Ensemble: aggregate the per-seed checkpoints from train_multiseed.py.
                   Disagreement across independently-trained models = epistemic uncertainty.

For each, we report the mean predictive std (kW and % of mean power) and the empirical
95% coverage (fraction of true values within mean +/- 1.96*std) as a calibration check.

Usage:
    python evaluate_uncertainty.py                       # MC-Dropout on the main checkpoint
    python evaluate_uncertainty.py --mc-samples 50
    python evaluate_uncertainty.py --ensemble            # also run the deep ensemble
"""
import argparse
import glob

import numpy as np
import torch

from main_PI_NODE_Propeller import PINODEPropellerModel
from pinode_common import load_danae_temporal_sequences, make_loaders, predict_power, rmse_mape


def build_model(proc, feature_indices, calm_idx, weather_idx, input_size):
    # Canonical PI-NODE architecture — shared by both the MC-Dropout and ensemble paths.
    return PINODEPropellerModel(
        input_size=input_size,
        feature_indices=feature_indices,
        calm_water_indices=calm_idx,
        weather_indices=weather_idx,
        data_processor=proc,
        hidden_size=64,
        ode_num_layers=2,
        loss_function_choice="SmoothL1",
        encoder_mode="first",
    )


def load_into(model, path):
    # weights_only=True avoids arbitrary-code execution in pickled checkpoints.
    state = torch.load(path, map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()
    return model


def coverage(mean, std, true, z=1.96):
    """Fraction of true values within mean +/- z*std (perfectly calibrated 95% ~ 0.95)."""
    # np.maximum guards against zero-std edge cases that would inflate coverage artificially.
    within = np.abs(true - mean) <= z * np.maximum(std, 1e-9)
    return float(within.mean())


def report(tag, mean, std, true):
    # Summarise a single uncertainty estimator in three complementary numbers:
    # point-prediction quality (RMSE/MAPE), spread (mean std), and interval calibration (coverage).
    rmse, mape = rmse_mape(mean, true)
    mean_std = float(std.mean())
    # Express spread relative to mean predicted power so it is dimensionless and comparable across ships.
    rel = mean_std / max(float(np.abs(mean).mean()), 1e-9) * 100
    cov = coverage(mean, std, true)
    print(f"\n--- {tag} ---")
    print(f"  RMSE of mean prediction : {rmse:.2f} kW  (MAPE {mape:.2f}%)")
    print(f"  Mean predictive std     : {mean_std:.2f} kW  ({rel:.2f}% of mean power)")
    print(f"  Empirical 95% coverage  : {cov*100:.1f}%  (well-calibrated ~ 95%)")
    return {"tag": tag, "rmse_kw": round(rmse, 2), "mape_pct": round(mape, 3),
            "mean_std_kw": round(mean_std, 2), "coverage_95_pct": round(cov * 100, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_model_PI_NODE_danae.pt")
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--ensemble", action="store_true",
                    help="Also run the deep-ensemble UQ from results/best_model_PI_NODE_seed*.pt")
    ap.add_argument("--ensemble-glob", default="results/best_model_PI_NODE_seed*.pt")
    args = ap.parse_args()

    proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple = \
        load_danae_temporal_sequences()
    input_size = train_tuple[0].shape[2]

    # A throwaway model just to build the test loader (loaders are model-agnostic here).
    base = build_model(proc, feature_indices, calm_idx, weather_idx, input_size)
    _, _, test_loader = make_loaders(base, train_tuple, test_tuple)

    print("=" * 70)
    print("PI-NODE Uncertainty Quantification (DANAE test set)")
    print("=" * 70)

    # --- MC-Dropout on the single main checkpoint ---
    # Dropout is left active at inference time; each of the K passes samples a different
    # sub-network, so the spread of predictions approximates posterior predictive variance.
    model = load_into(base, args.checkpoint)
    runs = []
    for k in range(args.mc_samples):
        preds, true = predict_power(model, test_loader, mc_dropout=True)
        runs.append(preds)
    runs = np.stack(runs)  # (K, N)
    report(f"MC-Dropout (K={args.mc_samples})", runs.mean(0), runs.std(0), true)

    # --- Deep ensemble (optional; needs per-seed checkpoints) ---
    # Each member was trained from a different random seed, so inter-member disagreement
    # captures the epistemic uncertainty that a single MC-Dropout model cannot separate from
    # aleatoric noise.
    if args.ensemble:
        paths = sorted(glob.glob(args.ensemble_glob))
        if len(paths) < 2:
            print(f"\n[ensemble] Need >=2 checkpoints matching '{args.ensemble_glob}' "
                  f"(found {len(paths)}). Run train_multiseed.py first. Skipping.")
        else:
            ens_runs, true = [], None
            for p in paths:
                m = build_model(proc, feature_indices, calm_idx, weather_idx, input_size)
                m = load_into(m, p)
                preds, true = predict_power(m, test_loader)
                ens_runs.append(preds)
            ens_runs = np.stack(ens_runs)  # (M, N)
            report(f"Deep Ensemble (M={len(paths)})", ens_runs.mean(0), ens_runs.std(0), true)

    print("\nDone.")


if __name__ == "__main__":
    main()
