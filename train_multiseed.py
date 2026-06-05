"""Train the full PI-NODE under N random seeds for statistical significance.

Produces a mean +/- std of test RMSE/MAPE (answers "was the result a lucky seed?")
and saves one checkpoint per seed, which double as the members of the deep-ensemble
used by evaluate_uncertainty.py.

Usage:
    python train_multiseed.py                  # 5 seeds, full protocol
    python train_multiseed.py --seeds 0 1 2    # specific seeds
    python train_multiseed.py --epochs 100     # quicker pass
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from main_PI_NODE_Propeller import PINODEPropellerModel
from pinode_common import (
    load_danae_temporal_sequences, make_loaders, predict_power, rmse_mape,
)

LAMBDA_RANGE, LAMBDA_CURV, LAMBDA_PRIOR = 0.25, 0.01, 0.001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "multiseed_results.csv"

    # Resume: keep seeds already recorded in the CSV and skip re-running them.
    rows = []
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        done = {int(r["seed"]) for r in rows}
        if done:
            print(f"Resuming — seeds already done: {sorted(done)}; will skip these.")

    # Data/sequences are identical across seeds — load once.
    proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple = \
        load_danae_temporal_sequences()

    for seed in args.seeds:
        if seed in done:
            print(f"[skip] seed {seed} already in {out_csv}")
            continue
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        model = PINODEPropellerModel(
            input_size=train_tuple[0].shape[2],
            feature_indices=feature_indices,
            calm_water_indices=calm_idx,
            weather_indices=weather_idx,
            data_processor=proc,
            hidden_size=64,
            ode_num_layers=2,
            lr=0.001,
            epochs=args.epochs,
            batch_size=64,
            loss_function_choice="SmoothL1",
            encoder_mode="first",
            seed=seed,
        )
        model.LAMBDA_KQ_RANGE = LAMBDA_RANGE
        model.LAMBDA_KQ_CURVATURE = LAMBDA_CURV
        model.LAMBDA_KQ_PRIOR = LAMBDA_PRIOR

        train_loader, val_loader, test_loader = make_loaders(model, train_tuple, test_tuple)

        ckpt = str(results_dir / f"best_model_PI_NODE_seed{seed}.pt")
        model.train(train_loader, val_loader=val_loader, checkpoint_path=ckpt)

        preds, true = predict_power(model, test_loader)
        rmse, mape = rmse_mape(preds, true)
        print(f"\n[seed {seed}] Test RMSE = {rmse:.2f} kW | MAPE = {mape:.2f}%")
        rows.append({"seed": seed, "test_rmse_kw": round(rmse, 2), "test_mape_pct": round(mape, 3)})

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    rmses = np.array([float(r["test_rmse_kw"]) for r in rows])
    mapes = np.array([float(r["test_mape_pct"]) for r in rows])
    print("\n" + "=" * 70)
    print(f"MULTI-SEED SUMMARY ({len(rows)} seeds)")
    print("=" * 70)
    print(f"  RMSE: {rmses.mean():.2f} +/- {rmses.std(ddof=1):.2f} kW")
    print(f"  MAPE: {mapes.mean():.3f} +/- {mapes.std(ddof=1):.3f} %")
    print(f"\nSaved -> {out_csv} (+ per-seed checkpoints for the deep ensemble)")


if __name__ == "__main__":
    main()
