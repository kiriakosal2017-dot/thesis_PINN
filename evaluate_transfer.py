"""Phase 4: Zero-Shot Transfer Learning.

Loads models trained ONLY on DANAE, swaps physical constants to the target ship,
and evaluates on the target ship's data WITHOUT any retraining.

For DATA and HYBRID: the neural network weights are frozen, the scaler from DANAE is used.
For PI-NODE: the neural network weights are frozen, but the propeller constants
(D, pitch ratio, etc.) are swapped to match the target ship via the .env file.

Usage:
    # Copy target ship's .env and run:
    cp .env.kastor .env
    python -u evaluate_transfer.py
"""
import os
import pickle
import numpy as np
import torch
from dotenv import load_dotenv

from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig, PropellerConfig
from main_DATA import DataDrivenModel
from main_HYBRID import UnifiedPhysicsHybridModel
from main_PI_NODE_Propeller import PINODEPropellerModel


def get_ship_name():
    """Infer the target ship name from DATA_FILE_PATH in .env."""
    path = DataConfig.FILE_PATH
    name = os.path.basename(path).replace("_Synchronized_usable_data.xlsx", "")
    return name


def align_features_to_danae(X_target_unscaled, danae_feature_names):
    """Align target ship's features to match DANAE's schema.
    Missing columns are filled with 0 (which becomes the scaler mean after standardization).
    Extra columns in target are dropped.
    """
    import pandas as pd
    aligned = pd.DataFrame(0.0, index=X_target_unscaled.index, columns=danae_feature_names)
    common = [c for c in danae_feature_names if c in X_target_unscaled.columns]
    missing = [c for c in danae_feature_names if c not in X_target_unscaled.columns]
    aligned[common] = X_target_unscaled[common]
    if missing:
        print(f"  Features missing in target (filled with 0): {missing}")
    return aligned


def load_target_data():
    """Load the target ship's data using current .env settings."""
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    # Tabular (for DATA / HYBRID)
    proc_tab = DataProcessor()
    res_tab = proc_tab.load_and_prepare_data()
    if res_tab is None:
        raise RuntimeError("Failed to load tabular data for target ship")
    X_train_tab, X_test_tab, X_train_uns_tab, X_test_uns_tab, y_train_tab, y_test_tab, _, _ = res_tab

    # Temporal (for PI-NODE)
    proc_temp = DataProcessor()
    res_temp = proc_temp.load_and_prepare_temporal_data()
    if res_temp is None:
        raise RuntimeError("Failed to load temporal data for target ship")
    X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t, _, _ = res_temp

    return (proc_tab, X_test_tab, X_test_uns_tab, y_test_tab,
            proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_test_t)


def evaluate_data_zeroshot(proc_tab_target, X_test_uns_tab, y_test_tab):
    """Zero-shot DATA model: load DANAE weights, evaluate on target ship."""
    print("\n--- DATA (Zero-Shot) ---")

    with open("data_processor_danae.pkl", "rb") as f:
        proc_danae = pickle.load(f)

    danae_features = list(proc_danae.scaler_X.feature_names_in_)
    input_size = len(danae_features)

    # Align target features to DANAE schema
    X_aligned = align_features_to_danae(X_test_uns_tab, danae_features)

    model = DataDrivenModel(input_size=input_size, hidden_layers=[128, 64, 32])
    state = torch.load("best_model_DATA_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()

    X_test_scaled = proc_danae.scaler_X.transform(X_aligned)
    X_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)

    with torch.no_grad():
        preds_scaled = model.model(X_t).cpu().numpy()

    preds_kw = proc_danae.inverse_transform_y(preds_scaled)
    true_kw = proc_tab_target.inverse_transform_y(y_test_tab.values.reshape(-1, 1))

    rmse = np.sqrt(np.mean((preds_kw.reshape(-1) - true_kw.reshape(-1)) ** 2))
    mape = np.mean(np.abs((preds_kw.reshape(-1) - true_kw.reshape(-1)) / np.maximum(np.abs(true_kw.reshape(-1)), 100.0))) * 100
    print(f"  DATA Zero-Shot RMSE: {rmse:.2f} kW, MAPE: {mape:.2f}%")
    return {"model": "DATA", "rmse": rmse, "mape": mape}


def evaluate_hybrid_zeroshot(proc_tab_target, X_test_uns_tab, y_test_tab):
    """Zero-shot HYBRID model: same as DATA at inference (physics only affects training)."""
    print("\n--- HYBRID (Zero-Shot) ---")

    with open("data_processor_danae.pkl", "rb") as f:
        proc_danae = pickle.load(f)

    danae_features = list(proc_danae.scaler_X.feature_names_in_)
    input_size = len(danae_features)

    X_aligned = align_features_to_danae(X_test_uns_tab, danae_features)

    model = UnifiedPhysicsHybridModel(input_size=input_size, hidden_layers=[128, 64, 32])
    state = torch.load("best_model_HYBRID_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()

    X_test_scaled = proc_danae.scaler_X.transform(X_aligned)
    X_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)

    with torch.no_grad():
        preds_scaled = model.model(X_t).cpu().numpy()

    preds_kw = proc_danae.inverse_transform_y(preds_scaled)
    true_kw = proc_tab_target.inverse_transform_y(y_test_tab.values.reshape(-1, 1))

    rmse = np.sqrt(np.mean((preds_kw.reshape(-1) - true_kw.reshape(-1)) ** 2))
    mape = np.mean(np.abs((preds_kw.reshape(-1) - true_kw.reshape(-1)) / np.maximum(np.abs(true_kw.reshape(-1)), 100.0))) * 100
    print(f"  HYBRID Zero-Shot RMSE: {rmse:.2f} kW, MAPE: {mape:.2f}%")
    return {"model": "HYBRID", "rmse": rmse, "mape": mape}


def evaluate_pinode_zeroshot(proc_temp_target, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_test_t):
    """Zero-shot PI-NODE: load DANAE NN weights but use TARGET ship's propeller constants."""
    print("\n--- PI-NODE (Zero-Shot) ---")
    print(f"  Target propeller: D={PropellerConfig.D}m, Z={PropellerConfig.Z}, P/D={PropellerConfig.P_D}")

    with open("data_processor_danae_temporal.pkl", "rb") as f:
        proc_danae = pickle.load(f)

    # Use DANAE's feature schema (the model was trained with these exact features)
    danae_features = list(proc_danae.scaler_X.feature_names_in_)
    feature_indices = {c: i for i, c in enumerate(danae_features)}

    calm_water_cols = [
        col for col in danae_features
        if not any(w in col.lower() for w in ['wind', 'wave', 'swell'])
    ]
    weather_cols = [
        col for col in danae_features
        if any(w in col.lower() for w in ['wind', 'wave', 'swell'])
    ]
    calm_water_indices = [feature_indices[col] for col in calm_water_cols]
    weather_indices = [feature_indices[col] for col in weather_cols]

    input_size = len(danae_features)

    # Build model with DANAE's architecture but TARGET ship's propeller config
    model = PINODEPropellerModel(
        input_size=input_size,
        feature_indices=feature_indices,
        calm_water_indices=calm_water_indices,
        weather_indices=weather_indices,
        data_processor=proc_danae,
        hidden_size=64, ode_num_layers=2,
        loss_function_choice="SmoothL1",
        encoder_mode="first",
    )

    state = torch.load("best_model_PI_NODE_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)
    model.model.eval()

    # Align target unscaled data to DANAE schema
    import pandas as pd
    X_test_uns_aligned = align_features_to_danae(X_test_uns_t, danae_features)

    # Scale with DANAE's scaler
    X_test_scaled = proc_danae.scaler_X.transform(X_test_uns_aligned)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=danae_features, index=X_test_uns_aligned.index)

    seq_len = SequenceConfig.LENGTH
    X_te_seq_s, X_te_uns_seq_s, y_te_seq_s = create_sequences(
        X_test_scaled_df, X_test_uns_aligned, y_test_t, seq_length=seq_len
    )

    loader = model.prepare_sequence_dataloader(X_te_seq_s, X_te_uns_seq_s, y_te_seq_s, shuffle=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, X_uns_batch, y_batch in loader:
            X_batch = X_batch.to(model.device)
            X_uns_batch = X_uns_batch.to(model.device)
            w, t, eta_r = model.model(X_batch)
            X_uns_last = X_uns_batch[:, -1, :]
            P_kw, _, _, _, _ = model.compute_analytical_power(w, t, eta_r, X_uns_last)
            all_preds.append(P_kw.cpu().numpy())
            y_true_kw = proc_temp_target.inverse_transform_y(y_batch.numpy())
            all_true.append(y_true_kw)

    preds_kw = np.concatenate(all_preds).reshape(-1)
    true_kw = np.concatenate(all_true).reshape(-1)

    rmse = np.sqrt(np.mean((preds_kw - true_kw) ** 2))
    mape = np.mean(np.abs((preds_kw - true_kw) / np.maximum(np.abs(true_kw), 100.0))) * 100
    print(f"  PI-NODE Zero-Shot RMSE: {rmse:.2f} kW, MAPE: {mape:.2f}%")
    return {"model": "PI-NODE", "rmse": rmse, "mape": mape}


def main():
    ship_name = get_ship_name()
    print(f"Phase 4: Zero-Shot Transfer Learning → {ship_name}")
    print(f"Models trained on DANAE, evaluated on {ship_name} (no retraining)\n")

    (proc_tab, X_test_tab, X_test_uns_tab, y_test_tab,
     proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_test_t) = load_target_data()

    results = []
    results.append(evaluate_data_zeroshot(proc_tab, X_test_uns_tab, y_test_tab))
    results.append(evaluate_hybrid_zeroshot(proc_tab, X_test_uns_tab, y_test_tab))
    results.append(evaluate_pinode_zeroshot(proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_test_t))

    print("\n" + "=" * 70)
    print(f"PHASE 4 SUMMARY: Zero-Shot Transfer → {ship_name}")
    print(f"(All models trained on DANAE, no retraining on {ship_name})")
    print("=" * 70)
    print(f"  {'Model':<12} {'RMSE (kW)':>12} {'MAPE':>10}")
    print(f"  {'-' * 36}")
    for r in results:
        print(f"  {r['model']:<12} {r['rmse']:>10.2f}kW {r['mape']:>9.2f}%")

    print(f"\n  Reference (trained ON DANAE):")
    print(f"  {'DATA':<12} {'424.80':>10}kW {'5.73':>9}%")
    print(f"  {'HYBRID':<12} {'703.69':>10}kW {'8.88':>9}%")
    print(f"  {'PI-NODE':<12} {'275.95':>10}kW {'3.10':>9}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
