"""Train the PI-KAN baseline under N seeds for a confidence interval.

Mirrors train_multiseed.py's resume-from-CSV pattern but uses the tabular pipeline
(identical to HYBRID) and PIKANModel, producing results/multiseed_pikan_results.csv
for a direct apples-to-apples comparison with the PI-NODE multi-seed numbers.

Usage:
    python train_multiseed_pikan.py                  # seeds 0..4, full protocol
    python train_multiseed_pikan.py --seeds 0 1 2    # specific seeds
    python train_multiseed_pikan.py --epochs 100     # quicker pass
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from config import TrainingConfig
from main_HYBRID import _build_feature_indices
from main_PI_KAN import PIKANModel
from read_data import DataProcessor

# Architecture tail kept in sync with evaluate_pikan.py so checkpoints are
# loadable by the evaluation script without modification.
KAN_WIDTH_TAIL = [64, 32, 1]


def predict_kw(model, X, dp):
    """Run a full forward pass and invert the output scaler to physical kW."""
    model.model.eval()
    with torch.no_grad():
        xb = torch.tensor(X.values, dtype=torch.float32, device=model.device)
        out_scaled = model.model(xb).cpu().numpy()
    return dp.inverse_transform_y(out_scaled).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=TrainingConfig.EPOCHS_FINAL)
    args = ap.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "multiseed_pikan_results.csv"

    # Resume-from-CSV: read completed seeds and skip them on restart.
    rows, done = [], set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        done = {int(r["seed"]) for r in rows}
        if done:
            print(f"Resuming — seeds already done: {sorted(done)}; will skip these.")

    # Tabular preprocessing is deterministic and shared across all seeds;
    # load and scale once to avoid redundant I/O.
    dp = DataProcessor()
    result = dp.load_and_prepare_data()
    if result is None:
        raise RuntimeError("Failed to load data")
    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = result
    feature_indices = _build_feature_indices(X_train_uns)
    in_size = X_train.shape[1]

    # Chronological 80/20 split: the last 20% of training rows form the validation
    # set so that later time steps are never used to tune earlier ones.
    n_val = int(len(X_train) * 0.2)
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    X_tr_un = X_train_uns.iloc[:-n_val]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    # Invert test labels once here; avoids repeated scaler calls inside the loop.
    true = dp.inverse_transform_y(y_test.values).ravel()

    for seed in args.seeds:
        if seed in done:
            print(f"[skip] seed {seed} already in {out_csv}")
            continue
        print("\n" + "=" * 70 + f"\nSEED {seed}\n" + "=" * 70)

        # seed= is applied INSIDE the constructor (after BaseModel.__init__ resets the
        # RNG), so each seed genuinely differs. batch_size=512 + CPU (force_cpu default)
        # keep the physics-guided KAN training stable and numerically correct.
        model = PIKANModel(
            input_size=in_size,
            kan_width=[in_size] + KAN_WIDTH_TAIL,
            lr=0.001,
            epochs=args.epochs,
            batch_size=512,
            seed=seed,
        )
        train_loader = model.prepare_combined_dataloader(X_tr, X_tr_un, y_tr, shuffle=True)
        val_loader = model.prepare_dataloader(X_val, y_val)
        # Seed-specific checkpoint path preserves all N models for ensemble use.
        ckpt = str(results_dir / f"best_model_PI_KAN_seed{seed}.pt")
        model.train(train_loader, X_tr_un, feature_indices, dp,
                    val_loader=val_loader, checkpoint_path=ckpt)

        preds = predict_kw(model, X_test, dp)
        rmse = float(np.sqrt(np.mean((preds - true) ** 2)))
        mape = float(np.mean(np.abs((preds - true) / np.maximum(np.abs(true), 100.0))) * 100)
        print(f"\n[seed {seed}] Test RMSE = {rmse:.2f} kW | MAPE = {mape:.2f}%")
        rows.append({"seed": seed, "test_rmse_kw": round(rmse, 2),
                     "test_mape_pct": round(mape, 3)})

        # Write incrementally so only the in-flight seed is lost on a crash.
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    rmses = np.array([float(r["test_rmse_kw"]) for r in rows])
    mapes = np.array([float(r["test_mape_pct"]) for r in rows])
    print("\n" + "=" * 70 + f"\nPI-KAN MULTI-SEED SUMMARY ({len(rows)} seeds)\n" + "=" * 70)
    # ddof=1: unbiased sample std over the small seed ensemble.
    print(f"  RMSE: {rmses.mean():.2f} +/- {rmses.std(ddof=1):.2f} kW")
    print(f"  MAPE: {mapes.mean():.3f} +/- {mapes.std(ddof=1):.3f} %")
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
