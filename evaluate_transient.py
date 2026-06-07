"""Split the DANAE test set into steady and transient regimes by |dV/dt| and
compare DATA, HYBRID, and PI-NODE error on each subset to assess whether physics
inductive bias improves robustness during non-equilibrium operating conditions."""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from read_data import DataProcessor, create_sequences, split_calm_weather_indices
from config import DataConfig, SequenceConfig, ColumnConfig
from main_DATA import DataDrivenModel
from main_HYBRID import UnifiedPhysicsHybridModel
from main_PI_NODE_Propeller import PINODEPropellerModel


def load_data():
    """Load both tabular (for DATA/HYBRID) and temporal (for PI-NODE) data."""
    # Tabular branch: flat feature vectors; acceleration/dt columns are absent by design.
    proc_tab = DataProcessor()
    res_tab = proc_tab.load_and_prepare_data()
    if res_tab is None:
        raise RuntimeError("Failed to load tabular data")
    X_train_tab, X_test_tab, _, _, y_train_tab, y_test_tab, _, _ = res_tab

    # RPM is normally dropped to prevent leakage; the temporal branch needs it
    # because the ODE integrator uses rotational speed as a state variable.
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    # Temporal branch: includes dt and acceleration columns used by PI-NODE sequences.
    proc_temp = DataProcessor()
    res_temp = proc_temp.load_and_prepare_temporal_data()
    if res_temp is None:
        raise RuntimeError("Failed to load temporal data")
    X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t, _, _ = res_temp

    return (proc_tab, X_train_tab, X_test_tab, y_train_tab, y_test_tab,
            proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t)


def identify_transient_indices(X_test_uns, percentile=75):
    """Label each test row as transient or steady-state based on |dV/dt|.

    The threshold is data-driven (a percentile of the test set's acceleration
    distribution) rather than a fixed physical constant, which keeps the
    comparison balanced across different vessel speed ranges.
    """
    accel_col = "acceleration"
    if accel_col not in X_test_uns.columns:
        raise ValueError("acceleration column not found in unscaled test data")

    accel = X_test_uns[accel_col].values
    abs_accel = np.abs(accel)
    # Rows at or above this percentile are treated as transient; the rest are steady.
    threshold = np.percentile(abs_accel, percentile)

    transient_mask = abs_accel >= threshold
    print(f"Transient threshold (P{percentile}): |accel| >= {threshold:.6f} m/s^2")
    print(f"  Steady-state samples: {(~transient_mask).sum()}")
    print(f"  Transient samples:    {transient_mask.sum()}")
    return transient_mask, threshold


def evaluate_data_model(X_test_tab, y_test_tab, proc_tab, transient_mask):
    """Evaluate the saved DATA model on transient and steady subsets."""
    print("\n--- DATA Model ---")
    input_size = X_test_tab.shape[1]
    model = DataDrivenModel(input_size=input_size, hidden_layers=[128, 64, 32])
    state = torch.load("best_model_DATA_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()

    X_t = torch.tensor(X_test_tab.values, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        preds_scaled = model.model(X_t).cpu().numpy()

    preds_kw = proc_tab.inverse_transform_y(preds_scaled)
    true_kw = proc_tab.inverse_transform_y(y_test_tab.values.reshape(-1, 1))

    # The tabular and temporal processors may yield slightly different row counts
    # (e.g. due to NaN dropping); trim or pad the mask accordingly.
    mask = transient_mask[:len(preds_kw)] if len(transient_mask) >= len(preds_kw) else np.pad(transient_mask, (0, len(preds_kw) - len(transient_mask)), constant_values=False)

    return compute_rmse_split(preds_kw, true_kw, mask, "DATA")


def evaluate_hybrid_model(X_test_tab, y_test_tab, proc_tab, transient_mask):
    """Evaluate the saved HYBRID model on transient and steady subsets.

    At inference, HYBRID is architecturally identical to DATA — the physics
    regularisation only alters the training loss, not the forward pass.
    """
    print("\n--- HYBRID Model ---")
    input_size = X_test_tab.shape[1]
    model = UnifiedPhysicsHybridModel(input_size=input_size, hidden_layers=[128, 64, 32])
    state = torch.load("best_model_HYBRID_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()

    X_t = torch.tensor(X_test_tab.values, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        preds_scaled = model.model(X_t).cpu().numpy()

    preds_kw = proc_tab.inverse_transform_y(preds_scaled)
    true_kw = proc_tab.inverse_transform_y(y_test_tab.values.reshape(-1, 1))

    mask = transient_mask[:len(preds_kw)] if len(transient_mask) >= len(preds_kw) else np.pad(transient_mask, (0, len(preds_kw) - len(transient_mask)), constant_values=False)

    return compute_rmse_split(preds_kw, true_kw, mask, "HYBRID")


def evaluate_pinode_model(proc, X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, transient_mask):
    """Evaluate the saved PI-NODE model on transient and steady subsets."""
    print("\n--- PI-NODE Model ---")
    feature_indices = {c: i for i, c in enumerate(X_train.columns)}

    calm_water_indices, weather_indices = split_calm_weather_indices(X_train.columns)

    seq_len = SequenceConfig.LENGTH
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(
        X_test, X_test_uns, y_test, seq_length=seq_len
    )

    model = PINODEPropellerModel(
        input_size=X_te_seq.shape[2],
        feature_indices=feature_indices,
        calm_water_indices=calm_water_indices,
        weather_indices=weather_indices,
        data_processor=proc,
        hidden_size=64, ode_num_layers=2,
        loss_function_choice="SmoothL1",
        encoder_mode="first",
    )
    state = torch.load("best_model_PI_NODE_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()

    # shuffle=False is mandatory: the transient mask is positionally aligned to the
    # original time-sorted test set, so reordering rows would corrupt the split.
    loader = model.prepare_sequence_dataloader(X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, X_uns_batch, y_batch in loader:
            X_batch = X_batch.to(model.device)
            X_uns_batch = X_uns_batch.to(model.device)
            w, t, eta_r = model.model(X_batch)
            # The analytical power formula reads physical quantities (speed, density)
            # from the last timestep of each window (the prediction target's instant).
            X_uns_last = X_uns_batch[:, -1, :]
            P_kw, _, _, _, _ = model.compute_analytical_power(w, t, eta_r, X_uns_last)
            all_preds.append(P_kw.cpu().numpy())
            y_true_kw = proc.inverse_transform_y(y_batch.numpy())
            all_true.append(y_true_kw)

    preds_kw = np.concatenate(all_preds).reshape(-1)
    true_kw = np.concatenate(all_true).reshape(-1)

    # Sliding windows consume the first (seq_len - 1) rows as context, so prediction
    # index 0 corresponds to original row (seq_len - 1).  Offset the mask to match.
    seq_mask = transient_mask[seq_len - 1:]
    if len(seq_mask) > len(preds_kw):
        seq_mask = seq_mask[:len(preds_kw)]
    elif len(seq_mask) < len(preds_kw):
        preds_kw = preds_kw[:len(seq_mask)]
        true_kw = true_kw[:len(seq_mask)]

    return compute_rmse_split(preds_kw, true_kw, seq_mask, "PI-NODE")


def compute_rmse_split(preds, true, mask, model_name):
    """Compute RMSE and MAPE for overall, steady-state, and transient subsets."""
    preds = preds.reshape(-1)
    true = true.reshape(-1)

    rmse_all = np.sqrt(np.mean((preds - true) ** 2))
    rmse_steady = np.sqrt(np.mean((preds[~mask] - true[~mask]) ** 2))
    rmse_transient = np.sqrt(np.mean((preds[mask] - true[mask]) ** 2))

    # Clamp the denominator to 100 kW to prevent division instability at near-zero
    # power values (e.g. during slow manoeuvring), which would inflate MAPE artificially.
    safe_true = np.where(np.abs(true) > 100, true, 100.0)
    mape_all = np.mean(np.abs((preds - true) / safe_true)) * 100
    mape_steady = np.mean(np.abs((preds[~mask] - true[~mask]) / safe_true[~mask])) * 100
    mape_transient = np.mean(np.abs((preds[mask] - true[mask]) / safe_true[mask])) * 100

    print(f"  {model_name} Overall:    RMSE={rmse_all:.2f} kW,  MAPE={mape_all:.2f}%")
    print(f"  {model_name} Steady:     RMSE={rmse_steady:.2f} kW,  MAPE={mape_steady:.2f}%  ({(~mask).sum()} samples)")
    print(f"  {model_name} Transient:  RMSE={rmse_transient:.2f} kW,  MAPE={mape_transient:.2f}%  ({mask.sum()} samples)")

    return {
        "model": model_name,
        "overall": rmse_all, "steady": rmse_steady, "transient": rmse_transient,
        "mape_overall": mape_all, "mape_steady": mape_steady, "mape_transient": mape_transient,
    }


def run_evaluation_at_percentile(percentile, proc_tab, X_test_tab, y_test_tab,
                                  proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t):
    """Run full evaluation for a given transient percentile threshold."""
    transient_mask, threshold = identify_transient_indices(X_test_uns_t, percentile=percentile)

    results = []
    results.append(evaluate_data_model(X_test_tab, y_test_tab, proc_tab, transient_mask))
    results.append(evaluate_hybrid_model(X_test_tab, y_test_tab, proc_tab, transient_mask))
    results.append(evaluate_pinode_model(
        proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t, transient_mask
    ))

    print("\n" + "=" * 70)
    print(f"RESULTS at P{percentile}: |dV/dt| >= {threshold:.6f} m/s^2")
    print("=" * 70)
    print(f"{'Model':<12} {'Overall (kW)':>14} {'Steady (kW)':>14} {'Transient (kW)':>16}")
    print("-" * 58)
    for r in results:
        print(f"{r['model']:<12} {r['overall']:>14.2f} {r['steady']:>14.2f} {r['transient']:>16.2f}")
    print("=" * 70)
    return results, threshold


def main():
    (proc_tab, X_train_tab, X_test_tab, y_train_tab, y_test_tab,
     proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t) = load_data()

    all_results = {}
    for pct in [75, 90]:
        print(f"\n{'#' * 70}")
        print(f"  TRANSIENT ANALYSIS at P{pct}")
        print(f"{'#' * 70}")
        results, threshold = run_evaluation_at_percentile(
            pct, proc_tab, X_test_tab, y_test_tab,
            proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t
        )
        all_results[pct] = (results, threshold)

    print("\n" + "=" * 80)
    print("Steady vs transient regime summary")
    print("=" * 80)
    for pct, (results, threshold) in all_results.items():
        print(f"\n  P{pct} threshold: |dV/dt| >= {threshold:.6f} m/s^2")
        print(f"  {'Model':<12} {'Steady RMSE':>12} {'Trans RMSE':>12} {'Steady MAPE':>13} {'Trans MAPE':>12}")
        print(f"  {'-' * 62}")
        for r in results:
            print(f"  {r['model']:<12} {r['steady']:>10.2f}kW {r['transient']:>10.2f}kW {r['mape_steady']:>11.2f}% {r['mape_transient']:>10.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
