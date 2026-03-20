import torch
import numpy as np
from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig
from torch.utils.data import TensorDataset, DataLoader

from main_DATA import DataDrivenModel
from main_HYBRID import UnifiedPhysicsHybridModel

def main():
    print("Loading data without leakage columns...")
    proc_tabular = DataProcessor()
    res_tab = proc_tabular.load_and_prepare_data()
    if res_tab is None:
        raise RuntimeError("Failed to load tabular data")
        
    X_train_tab, X_test_tab, X_train_uns_tab, X_test_uns_tab, y_train_tab, y_test_tab, y_train_uns_tab, y_test_uns_tab = res_tab
    
    # 1. Evaluate DATA Model
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
    
    # Validation split for early stopping
    n_val_tab = int(len(X_tr_tab_t) * 0.2)
    val_loader_tab = DataLoader(TensorDataset(X_tr_tab_t[-n_val_tab:], y_tr_tab_t[-n_val_tab:]), batch_size=64)
    train_loader_tab_sub = DataLoader(TensorDataset(X_tr_tab_t[:-n_val_tab], y_tr_tab_t[:-n_val_tab]), batch_size=64, shuffle=True)
    
    data_model.train(train_loader_tab_sub, val_loader=val_loader_tab, live_plot=False, checkpoint_path="best_model_DATA_danae.pt")
    
    print("\nEvaluating DATA Model on Test Set...")
    data_model.evaluate(X_test_tab, y_test_tab, dataset_type="Test", data_processor=proc_tabular)
    
    n_val_tab = int(len(X_train_tab) * 0.2)
    
    # 2. Evaluate HYBRID Model
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
    
    # Use pandas objects for BaseModel's prepare_dataloader
    X_tr_hybrid = X_train_tab.iloc[:-n_val_tab]
    y_tr_hybrid = y_train_tab.iloc[:-n_val_tab]
    X_tr_unscaled_hybrid = X_train_uns_tab.iloc[:-n_val_tab]
    
    X_val_hybrid = X_train_tab.iloc[-n_val_tab:]
    y_val_hybrid = y_train_tab.iloc[-n_val_tab:]
    
    train_loader_hybrid = hybrid_model.prepare_dataloader(X_tr_hybrid, y_tr_hybrid)
    unscaled_loader_hybrid = hybrid_model.prepare_unscaled_dataloader(X_tr_unscaled_hybrid)
    val_loader_hybrid = hybrid_model.prepare_dataloader(X_val_hybrid, y_val_hybrid)
    
    hybrid_feature_indices = {c: i for i, c in enumerate(X_train_tab.columns)}
    
    hybrid_model.train(
        train_loader=train_loader_hybrid,
        unscaled_loader=unscaled_loader_hybrid,
        X_train_unscaled=X_tr_unscaled_hybrid,
        feature_indices=hybrid_feature_indices,
        data_processor=proc_tabular,
        val_loader=val_loader_hybrid,
        checkpoint_path="best_model_HYBRID_danae.pt"
    )
    
    print("\nEvaluating HYBRID Model on Test Set...")
    hybrid_model.evaluate(X_test_tab, y_test_tab, dataset_type="Test", data_processor=proc_tabular)
    
    # Save the data processor (scaler) state as well so we can decode the predictions later
    import pickle
    with open("data_processor_danae.pkl", "wb") as f:
        pickle.dump(proc_tabular, f)
    print("Saved best_model_HYBRID_danae.pt and data_processor_danae.pkl for Transfer Learning!")


if __name__ == "__main__":
    main()
