"""Train a single PI-KAN on DANAE and report Test RMSE/MAPE (kW).

Mirrors the HYBRID path in evaluate_baselines.py: same DataProcessor split, same
features, chronological validation split, early stopping. Saves best_model_PI_KAN_danae.pt.
"""
import numpy as np
import torch

from config import DataConfig, TrainingConfig
from main_HYBRID import _build_feature_indices
from main_PI_KAN import PIKANModel
from read_data import DataProcessor

# Keep this width identical to Task 4 (train_multiseed_pikan.py).
KAN_WIDTH_TAIL = [64, 32, 1]


def predict_kw(model, X, data_processor):
    # Single-batch inference; avoids DataLoader overhead for a one-off evaluation pass.
    model.model.eval()
    with torch.no_grad():
        xb = torch.tensor(X.values, dtype=torch.float32, device=model.device)
        out_scaled = model.model(xb).cpu().numpy()
    return data_processor.inverse_transform_y(out_scaled).ravel()


def main():
    dp = DataProcessor()
    result = dp.load_and_prepare_data()
    if result is None:
        raise RuntimeError("Failed to load data")
    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = result
    feature_indices = _build_feature_indices(X_train_uns)

    in_size = X_train.shape[1]
    # Chronological 80/20 split — shuffling would leak future information into validation.
    n_val = int(len(X_train) * 0.2)
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    X_tr_un = X_train_uns.iloc[:-n_val]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    # Prepend input dimension to the width list; KAN_WIDTH_TAIL defines the hidden + output layers.
    model = PIKANModel(
        input_size=in_size,
        kan_width=[in_size] + KAN_WIDTH_TAIL,
        lr=0.001,
        epochs=TrainingConfig.EPOCHS_FINAL,
        batch_size=512,
        seed=DataConfig.RANDOM_STATE,
    )
    print(f"PI-KAN trainable params: {model.n_params()}")

    # train_loader carries both scaled features and unscaled features (needed by the physics loss).
    train_loader = model.prepare_combined_dataloader(X_tr, X_tr_un, y_tr, shuffle=True)
    val_loader = model.prepare_dataloader(X_val, y_val)
    model.train(
        train_loader, X_tr_un, feature_indices, dp,
        val_loader=val_loader, checkpoint_path="best_model_PI_KAN_danae.pt",
        history_csv="results/history/PI_KAN_danae.csv",
    )

    # Restore best checkpoint (saved by train()) for the final test evaluation.
    preds = predict_kw(model, X_test, dp)
    true = dp.inverse_transform_y(y_test.values).ravel()
    rmse = float(np.sqrt(np.mean((preds - true) ** 2)))
    # Floor denominator at 100 kW to keep MAPE stable near zero shaft power.
    mape = float(np.mean(np.abs((preds - true) / np.maximum(np.abs(true), 100.0))) * 100)
    print(f"\nPI-KAN Test RMSE = {rmse:.2f} kW | MAPE = {mape:.2f}%")


if __name__ == "__main__":
    main()
