"""Train and evaluate the DATA and HYBRID baselines on the source vessel, then persist
the best checkpoints and fitted data processor for downstream transfer evaluation."""

import torch
import numpy as np
from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig
from torch.utils.data import TensorDataset, DataLoader

from main_DATA import DataDrivenModel
from main_HYBRID import UnifiedPhysicsHybridModel

def main():
    # DataProcessor strips leakage columns (e.g. shaft power directly measured)
    # before fitting the scaler, so test metrics reflect a realistic deployment scenario.
    print("Loading data without leakage columns...")
    proc_tabular = DataProcessor()
    res_tab = proc_tabular.load_and_prepare_data()
    if res_tab is None:
        raise RuntimeError("Failed to load tabular data")

    X_train_tab, X_test_tab, X_train_uns_tab, X_test_uns_tab, y_train_tab, y_test_tab, y_train_uns_tab, y_test_uns_tab = res_tab

    # --- DATA model ---
    print("\n" + "="*50)
    print("Training DATA Model (No Data Leakage)")
    print("="*50)
    data_model = DataDrivenModel(
        input_size=X_train_tab.shape[1],
        hidden_layers=[128, 64, 32],
        lr=0.001,
        epochs=1000,
        batch_size=64
    )

    X_tr_tab_t = torch.tensor(X_train_tab.values, dtype=torch.float32)
    y_tr_tab_t = torch.tensor(y_train_tab.values, dtype=torch.float32).view(-1, 1)

    # Chronological 80/20 split: the last 20 % of the training period is held out for
    # early stopping.  Shuffling is intentionally absent here to avoid temporal leakage.
    n_val_tab = int(len(X_tr_tab_t) * 0.2)
    val_loader_tab = DataLoader(TensorDataset(X_tr_tab_t[-n_val_tab:], y_tr_tab_t[-n_val_tab:]), batch_size=64)
    train_loader_tab_sub = DataLoader(TensorDataset(X_tr_tab_t[:-n_val_tab], y_tr_tab_t[:-n_val_tab]), batch_size=64, shuffle=True)

    data_model.train(train_loader_tab_sub, val_loader=val_loader_tab, live_plot=False, checkpoint_path="best_model_DATA_danae.pt", history_csv="results/history/DATA_danae.csv")

    print("\nEvaluating DATA Model on Test Set...")
    data_model.evaluate(X_test_tab, y_test_tab, dataset_type="Test", data_processor=proc_tabular)

    # Recompute n_val_tab from the full DataFrame length so HYBRID's slice indices are
    # consistent with the same 80/20 boundary used above.
    n_val_tab = int(len(X_train_tab) * 0.2)

    # --- HYBRID model ---
    print("\n" + "="*50)
    print("Training HYBRID Model (No Data Leakage)")
    print("="*50)

    hybrid_model = UnifiedPhysicsHybridModel(
        input_size=X_train_tab.shape[1],
        hidden_layers=[128, 64, 32],
        lr=0.001,
        epochs=1000,
        batch_size=64
    )

    # HYBRID's training loop expects raw (unscaled) features alongside scaled ones so it
    # can evaluate the physics residual term; pandas DataFrames are passed because
    # BaseModel.prepare_dataloader relies on column names for that alignment.
    X_tr_hybrid = X_train_tab.iloc[:-n_val_tab]
    y_tr_hybrid = y_train_tab.iloc[:-n_val_tab]
    X_tr_unscaled_hybrid = X_train_uns_tab.iloc[:-n_val_tab]

    X_val_hybrid = X_train_tab.iloc[-n_val_tab:]
    y_val_hybrid = y_train_tab.iloc[-n_val_tab:]

    train_loader_hybrid = hybrid_model.prepare_combined_dataloader(
        X_tr_hybrid, X_tr_unscaled_hybrid, y_tr_hybrid, shuffle=True
    )
    val_loader_hybrid = hybrid_model.prepare_dataloader(X_val_hybrid, y_val_hybrid)

    hybrid_feature_indices = {c: i for i, c in enumerate(X_train_tab.columns)}

    hybrid_model.train(
        train_loader=train_loader_hybrid,
        X_train_unscaled=X_tr_unscaled_hybrid,
        feature_indices=hybrid_feature_indices,
        data_processor=proc_tabular,
        val_loader=val_loader_hybrid,
        checkpoint_path="best_model_HYBRID_danae.pt",
        history_csv="results/history/HYBRID_danae.csv"
    )

    print("\nEvaluating HYBRID Model on Test Set...")
    hybrid_model.evaluate(X_test_tab, y_test_tab, dataset_type="Test", data_processor=proc_tabular)

    # Persist the fitted scaler so the transfer script can apply the identical
    # source vessel normalisation to a target vessel's features without refitting.
    import pickle
    with open("data_processor_danae.pkl", "wb") as f:
        pickle.dump(proc_tabular, f)
    print("Saved best_model_HYBRID_danae.pt and data_processor_danae.pkl for Transfer Learning!")


if __name__ == "__main__":
    main()
