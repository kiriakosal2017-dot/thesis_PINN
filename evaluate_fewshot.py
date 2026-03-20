"""Phase 5: Few-Shot Fine-Tuning.

Loads models trained on DANAE, fine-tunes them on a small subset of target ship data
(e.g., 1 week, 1 month), and evaluates on the rest of the target ship's test set.

Demonstrates how quickly each model adapts to a new vessel with limited data.

Usage:
    cp .env.kastor .env
    python -u evaluate_fewshot.py
    cp .env.danae .env
"""
import os
import copy
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig, PropellerConfig
from main_DATA import DataDrivenModel
from main_HYBRID import UnifiedPhysicsHybridModel
from main_PI_NODE_Propeller import PINODEPropellerModel


FEWSHOT_FRACTIONS = [0.01, 0.05, 0.10, 0.25]
FEWSHOT_LR = 1e-4
FEWSHOT_EPOCHS = 50
FEWSHOT_PATIENCE = 15


def get_ship_name():
    path = DataConfig.FILE_PATH
    return os.path.basename(path).replace("_Synchronized_usable_data.xlsx", "")


def align_features_to_danae(X_target_unscaled, danae_feature_names):
    import pandas as pd
    aligned = pd.DataFrame(0.0, index=X_target_unscaled.index, columns=danae_feature_names)
    common = [c for c in danae_feature_names if c in X_target_unscaled.columns]
    aligned[common] = X_target_unscaled[common]
    return aligned


def load_target_data():
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    proc_tab = DataProcessor()
    res_tab = proc_tab.load_and_prepare_data()
    if res_tab is None:
        raise RuntimeError("Failed to load tabular data")
    X_train_tab, X_test_tab, X_train_uns_tab, X_test_uns_tab, y_train_tab, y_test_tab, _, _ = res_tab

    proc_temp = DataProcessor()
    res_temp = proc_temp.load_and_prepare_temporal_data()
    if res_temp is None:
        raise RuntimeError("Failed to load temporal data")
    X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t, _, _ = res_temp

    # Also capture unscaled y for correct fine-tuning
    _, _, _, _, _, _, y_train_uns_tab, y_test_uns_tab = res_tab
    _, _, _, _, _, _, y_train_uns_t, y_test_uns_t = res_temp

    return (proc_tab, X_train_tab, X_test_tab, X_train_uns_tab, X_test_uns_tab,
            y_train_tab, y_test_tab, y_train_uns_tab, y_test_uns_tab,
            proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t,
            y_train_t, y_test_t, y_train_uns_t, y_test_uns_t)


def finetune_data_model(frac, X_train_uns_tab, y_train_uns_tab, X_test_uns_tab, y_test_uns_tab, proc_tab):
    with open("data_processor_danae.pkl", "rb") as f:
        proc_danae = pickle.load(f)

    danae_features = list(proc_danae.scaler_X.feature_names_in_)
    input_size = len(danae_features)

    model = DataDrivenModel(input_size=input_size, hidden_layers=[128, 64, 32],
                            lr=FEWSHOT_LR, epochs=FEWSHOT_EPOCHS, batch_size=64)
    state = torch.load("best_model_DATA_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)

    n_ft = max(1, int(len(X_train_uns_tab) * frac))
    X_ft = align_features_to_danae(X_train_uns_tab.iloc[:n_ft], danae_features)
    X_ft_scaled = proc_danae.scaler_X.transform(X_ft)
    # Use UNSCALED y, then scale with DANAE's scaler
    y_ft_uns = y_train_uns_tab.iloc[:n_ft]
    y_ft_scaled = proc_danae.scaler_y.transform(y_ft_uns.values.reshape(-1, 1))

    X_t = torch.tensor(X_ft_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_ft_scaled, dtype=torch.float32).view(-1, 1)
    n_val = max(1, int(len(X_t) * 0.2))
    train_loader = DataLoader(TensorDataset(X_t[:-n_val], y_t[:-n_val]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_t[-n_val:], y_t[-n_val:]), batch_size=64)

    model.train(train_loader, val_loader=val_loader, live_plot=False)

    X_test_aligned = align_features_to_danae(X_test_uns_tab, danae_features)
    X_test_scaled = proc_danae.scaler_X.transform(X_test_aligned)
    X_te = torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        preds_scaled = model.model(X_te).cpu().numpy()

    preds_kw = proc_danae.inverse_transform_y(preds_scaled).reshape(-1)
    true_kw = y_test_uns_tab.values.reshape(-1)
    rmse = np.sqrt(np.mean((preds_kw - true_kw) ** 2))
    mape = np.mean(np.abs((preds_kw - true_kw) / np.maximum(np.abs(true_kw), 100.0))) * 100
    return rmse, mape, n_ft


def finetune_pinode_model(frac, X_train_t, X_train_uns_t, y_train_t, y_train_uns_t,
                          X_test_t, X_test_uns_t, y_test_t, y_test_uns_t, proc_temp):
    with open("data_processor_danae_temporal.pkl", "rb") as f:
        proc_danae = pickle.load(f)

    danae_features = list(proc_danae.scaler_X.feature_names_in_)
    feature_indices = {c: i for i, c in enumerate(danae_features)}

    calm_water_cols = [c for c in danae_features if not any(w in c.lower() for w in ['wind', 'wave', 'swell'])]
    weather_cols = [c for c in danae_features if any(w in c.lower() for w in ['wind', 'wave', 'swell'])]
    calm_water_indices = [feature_indices[c] for c in calm_water_cols]
    weather_indices = [feature_indices[c] for c in weather_cols]

    input_size = len(danae_features)
    model = PINODEPropellerModel(
        input_size=input_size, feature_indices=feature_indices,
        calm_water_indices=calm_water_indices, weather_indices=weather_indices,
        data_processor=proc_danae, hidden_size=64, ode_num_layers=2,
        lr=FEWSHOT_LR, epochs=FEWSHOT_EPOCHS, batch_size=64,
        loss_function_choice="SmoothL1", encoder_mode="first",
    )
    model.LAMBDA_KQ_RANGE = 0.25
    model.LAMBDA_KQ_CURVATURE = 0.01
    model.LAMBDA_KQ_PRIOR = 0.001

    state = torch.load("best_model_PI_NODE_danae.pt", map_location=model.device, weights_only=True)
    model.model.load_state_dict(state)

    import pandas as pd
    n_ft = max(1, int(len(X_train_t) * frac))
    X_ft_uns = align_features_to_danae(X_train_uns_t.iloc[:n_ft], danae_features)
    X_ft_scaled = pd.DataFrame(proc_danae.scaler_X.transform(X_ft_uns), columns=danae_features, index=X_ft_uns.index)
    # Use UNSCALED y, then scale with DANAE's scaler
    y_ft_uns = y_train_uns_t.iloc[:n_ft]
    y_ft_scaled = pd.Series(proc_danae.scaler_y.transform(y_ft_uns.values.reshape(-1, 1)).flatten(),
                            index=y_ft_uns.index)

    seq_len = SequenceConfig.LENGTH
    X_ft_seq, X_ft_uns_seq, y_ft_seq = create_sequences(X_ft_scaled, X_ft_uns, y_ft_scaled, seq_length=seq_len)

    if len(X_ft_seq) < 2:
        return None, None, n_ft

    n_val = max(1, int(len(X_ft_seq) * 0.2))
    train_loader = model.prepare_sequence_dataloader(X_ft_seq[:-n_val], X_ft_uns_seq[:-n_val], y_ft_seq[:-n_val], shuffle=True)
    val_loader = model.prepare_sequence_dataloader(X_ft_seq[-n_val:], X_ft_uns_seq[-n_val:], y_ft_seq[-n_val:], shuffle=False)

    model.train(train_loader, val_loader=val_loader, live_plot=False)

    X_te_uns = align_features_to_danae(X_test_uns_t, danae_features)
    X_te_scaled = pd.DataFrame(proc_danae.scaler_X.transform(X_te_uns), columns=danae_features, index=X_te_uns.index)
    y_te_scaled = pd.Series(proc_danae.scaler_y.transform(y_test_uns_t.values.reshape(-1, 1)).flatten(),
                            index=y_test_uns_t.index)
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(X_te_scaled, X_te_uns, y_te_scaled, seq_length=seq_len)

    _, rmse = model.evaluate_loader(
        model.prepare_sequence_dataloader(X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False)
    )

    all_preds, all_true = [], []
    model.model.eval()
    loader = model.prepare_sequence_dataloader(X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False)
    with torch.no_grad():
        for X_b, X_uns_b, y_b in loader:
            X_b, X_uns_b = X_b.to(model.device), X_uns_b.to(model.device)
            w, t, eta_r = model.model(X_b)
            P_kw, _, _, _, _ = model.compute_analytical_power(w, t, eta_r, X_uns_b[:, -1, :])
            all_preds.append(P_kw.cpu().numpy())
            # y_b was scaled with DANAE's scaler, so decode with DANAE's scaler
            all_true.append(proc_danae.inverse_transform_y(y_b.numpy()))

    preds = np.concatenate(all_preds).reshape(-1)
    true = np.concatenate(all_true).reshape(-1)
    rmse = np.sqrt(np.mean((preds - true) ** 2))
    mape = np.mean(np.abs((preds - true) / np.maximum(np.abs(true), 100.0))) * 100
    return rmse, mape, n_ft


def main():
    ship_name = get_ship_name()
    print(f"Phase 5: Few-Shot Fine-Tuning → {ship_name}")
    print(f"Models pre-trained on DANAE, fine-tuned on small subsets of {ship_name}\n")

    (proc_tab, X_train_tab, X_test_tab, X_train_uns_tab, X_test_uns_tab,
     y_train_tab, y_test_tab, y_train_uns_tab, y_test_uns_tab,
     proc_temp, X_train_t, X_test_t, X_train_uns_t, X_test_uns_t,
     y_train_t, y_test_t, y_train_uns_t, y_test_uns_t) = load_target_data()

    print(f"Total training samples available: {len(X_train_tab)} (tabular), {len(X_train_t)} (temporal)")

    results = []
    for frac in FEWSHOT_FRACTIONS:
        print(f"\n{'='*70}")
        print(f"  Few-Shot: {frac*100:.0f}% of training data")
        print(f"{'='*70}")

        print(f"\n  --- DATA ({frac*100:.0f}%) ---")
        d_rmse, d_mape, d_n = finetune_data_model(frac, X_train_uns_tab, y_train_uns_tab, X_test_uns_tab, y_test_uns_tab, proc_tab)
        print(f"  DATA: RMSE={d_rmse:.2f} kW, MAPE={d_mape:.2f}% (n={d_n})")

        print(f"\n  --- PI-NODE ({frac*100:.0f}%) ---")
        p_rmse, p_mape, p_n = finetune_pinode_model(frac, X_train_t, X_train_uns_t, y_train_t, y_train_uns_t,
                                                      X_test_t, X_test_uns_t, y_test_t, y_test_uns_t, proc_temp)
        if p_rmse is not None:
            print(f"  PI-NODE: RMSE={p_rmse:.2f} kW, MAPE={p_mape:.2f}% (n={p_n})")
        else:
            print(f"  PI-NODE: Too few samples for sequences (n={p_n})")

        results.append({
            "frac": frac, "n_samples": d_n,
            "data_rmse": d_rmse, "data_mape": d_mape,
            "pinode_rmse": p_rmse, "pinode_mape": p_mape,
        })

    print(f"\n{'='*70}")
    print(f"PHASE 5 SUMMARY: Few-Shot Fine-Tuning → {ship_name}")
    print(f"{'='*70}")
    print(f"  {'Fraction':<10} {'Samples':>8} {'DATA MAPE':>12} {'PI-NODE MAPE':>14} {'Advantage':>12}")
    print(f"  {'-'*58}")
    for r in results:
        p_str = f"{r['pinode_mape']:.2f}%" if r['pinode_mape'] is not None else "N/A"
        adv = f"{r['data_mape']/r['pinode_mape']:.1f}x" if r['pinode_mape'] and r['pinode_mape'] > 0 else "N/A"
        print(f"  {r['frac']*100:>5.0f}%    {r['n_samples']:>8} {r['data_mape']:>10.2f}%   {p_str:>13} {adv:>12}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
