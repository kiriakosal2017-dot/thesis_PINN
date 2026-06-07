"""Ablation study for the PI-NODE architecture (source domain: DANAE).

Trains the full PI-NODE and several ablated variants under the identical protocol,
then reports each variant's test RMSE/MAPE so the contribution of each architectural
component can be isolated:

    full             : the complete PI-NODE (Neural ODE + weather branch + trainable B-series)
    no_ode           : Neural ODE removed -> GRU encoder decoded directly (recurrent baseline)
    no_weather       : Sea-State residual branch disabled
    frozen_bseries   : K_T/K_Q polynomial coefficients frozen at B-series init
    with_acceleration: acceleration re-added as an input feature (tests the transient confound)

Usage:
    python ablation_study.py                 # full protocol (1000 epochs, early stopping)
    python ablation_study.py --epochs 100    # quicker pass
    python ablation_study.py --configs full no_ode
"""
import argparse
import csv
from pathlib import Path

from config import DataConfig
from main_PI_NODE_Propeller import PINODEPropellerModel
from pinode_common import (
    load_danae_temporal_sequences, make_loaders, predict_power, rmse_mape,
)

# Best regularisation weights selected by sweep_pinode_regularization.py.
LAMBDA_RANGE, LAMBDA_CURV, LAMBDA_PRIOR = 0.25, 0.01, 0.001

# Each config specifies model flags and which meta columns to exclude from the feature matrix.
# Defaults preserve the full PI-NODE; only one knob changes per ablation row.
CONFIGS = {
    # Baseline: all three components active, acceleration excluded (it leaks transient effects).
    "full":              dict(use_ode=True,  use_weather=True,  freeze_polynomials=False, encoder_mode="first", meta_exclude=("dt", "acceleration")),
    # Replace the Neural ODE with a plain GRU decoder to isolate the ODE's contribution.
    "no_ode":            dict(use_ode=False, use_weather=True,  freeze_polynomials=False, encoder_mode="gru",   meta_exclude=("dt", "acceleration")),
    # Drop the sea-state residual branch to measure the weather correction's impact.
    "no_weather":        dict(use_ode=True,  use_weather=False, freeze_polynomials=False, encoder_mode="first", meta_exclude=("dt", "acceleration")),
    # Hold K_T/K_Q at their B-series initialisation to check whether learning them helps.
    "frozen_bseries":    dict(use_ode=True,  use_weather=True,  freeze_polynomials=True,  encoder_mode="first", meta_exclude=("dt", "acceleration")),
    # Re-introduce acceleration as a feature: if it hurts transfer it confirms the confound hypothesis.
    "with_acceleration": dict(use_ode=True,  use_weather=True,  freeze_polynomials=False, encoder_mode="first", meta_exclude=("dt",)),
}


def run_config(name, cfg, epochs, results_dir):
    print("\n" + "=" * 70)
    print(f"ABLATION: {name}  ({cfg})")
    print("=" * 70)

    # 'with_acceleration' needs a different feature split, so reload sequences for each config
    # rather than reusing a shared feature set.
    proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple = \
        load_danae_temporal_sequences(meta_exclude=cfg["meta_exclude"])

    model = PINODEPropellerModel(
        input_size=train_tuple[0].shape[2],
        feature_indices=feature_indices,
        calm_water_indices=calm_idx,
        weather_indices=weather_idx,
        data_processor=proc,
        hidden_size=64,
        ode_num_layers=2,
        lr=0.001,
        epochs=epochs,
        batch_size=64,
        loss_function_choice="SmoothL1",
        encoder_mode=cfg["encoder_mode"],
        use_ode=cfg["use_ode"],
        use_weather=cfg["use_weather"],
        freeze_polynomials=cfg["freeze_polynomials"],
    )
    # Apply the swept regularization strengths after construction (not exposed in __init__).
    model.LAMBDA_KQ_RANGE = LAMBDA_RANGE
    model.LAMBDA_KQ_CURVATURE = LAMBDA_CURV
    model.LAMBDA_KQ_PRIOR = LAMBDA_PRIOR

    train_loader, val_loader, test_loader = make_loaders(model, train_tuple, test_tuple)

    ckpt = str(results_dir / f"ablation_{name}.pt")
    model.train(train_loader, val_loader=val_loader, checkpoint_path=ckpt)

    preds, true = predict_power(model, test_loader)
    rmse, mape = rmse_mape(preds, true)
    print(f"\n[{name}] Test RMSE = {rmse:.2f} kW | MAPE = {mape:.2f}%")
    return {"config": name, "calm_features": len(calm_idx), "weather_features": len(weather_idx),
            "test_rmse_kw": round(rmse, 2), "test_mape_pct": round(mape, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000,
                    help="Max epochs per variant (early stopping still applies).")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                    help="Subset of configs to run.")
    args = ap.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "ablation_results.csv"

    # Resume: keep any configs already recorded in the CSV and skip re-running them.
    rows = []
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        done = {r["config"] for r in rows}
        if done:
            print(f"Resuming — already done: {sorted(done)}; will skip these.")

    for name in args.configs:
        if name not in CONFIGS:
            raise SystemExit(f"Unknown config '{name}'. Choices: {list(CONFIGS)}")
        if name in done:
            print(f"[skip] {name} already in {out_csv}")
            continue
        rows.append(run_config(name, CONFIGS[name], args.epochs, results_dir))

        # Write incrementally so partial progress survives an interruption.
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<20}{'RMSE (kW)':>12}{'MAPE (%)':>12}")
    for r in rows:
        print(f"  {r['config']:<20}{r['test_rmse_kw']:>12.2f}{r['test_mape_pct']:>12.3f}")
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
