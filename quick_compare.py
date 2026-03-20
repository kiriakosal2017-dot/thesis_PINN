import torch
import numpy as np
from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig

from main_DATA import DataDrivenModel
from main_PI_NODE_Propeller import PINODEPropellerModel

def main():
    print("Loading data without leakage columns...")
    proc_tabular = DataProcessor()
    res_tab = proc_tabular.load_and_prepare_data()
    X_train_tab, X_test_tab, _, _, y_train_tab, y_test_tab, _, y_test_uns_tab = res_tab
    
    proc_temp = DataProcessor()
    res_temp = proc_temp.load_and_prepare_temporal_data()
    X_train_t, X_test_t, X_train_uns_t, X_test_uns_t, y_train_t, y_test_t, _, y_test_uns_t = res_temp
    
    seq_len = SequenceConfig.LENGTH
    X_tr_seq, X_tr_uns_seq, y_tr_seq = create_sequences(X_train_t, X_train_uns_t, y_train_t, seq_length=seq_len)
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(X_test_t, X_test_uns_t, y_test_t, seq_length=seq_len)
    
    """
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
    
    from torch.utils.data import TensorDataset, DataLoader
    train_loader_tab = DataLoader(TensorDataset(X_tr_tab_t, y_tr_tab_t), batch_size=64, shuffle=True)
    
    # Validation split for early stopping
    n_val_tab = int(len(X_tr_tab_t) * 0.2)
    val_loader_tab = DataLoader(TensorDataset(X_tr_tab_t[-n_val_tab:], y_tr_tab_t[-n_val_tab:]), batch_size=64)
    train_loader_tab_sub = DataLoader(TensorDataset(X_tr_tab_t[:-n_val_tab], y_tr_tab_t[:-n_val_tab]), batch_size=64, shuffle=True)
    
    data_model.train(train_loader_tab_sub, val_loader=val_loader_tab, live_plot=False)
    
    print("\nEvaluating DATA Model on Test Set...")
    data_model.evaluate(X_test_tab, y_test_tab, dataset_type="Test", data_processor=proc_tabular)
    """
    
    """
    # 1.5 Evaluate HYBRID Model
    print("\n" + "="*50)
    print("Training HYBRID Model (No Data Leakage)")
    print("="*50)
    from main_HYBRID import UnifiedPhysicsHybridModel
    
    hybrid_model = UnifiedPhysicsHybridModel(
        input_size=X_train_tab.shape[1],
        hidden_layers=[128, 64, 32],
        lr=0.001,
        epochs=1000,
        batch_size=64
    )
    
    # HYBRID model needs unscaled inputs for physics equations
    X_tr_uns_tab_t = torch.tensor(X_train_uns_tab.values, dtype=torch.float32)
    train_loader_hybrid = DataLoader(TensorDataset(X_tr_tab_t, X_tr_uns_tab_t, y_tr_tab_t), batch_size=64, shuffle=True)
    
    val_loader_hybrid = DataLoader(TensorDataset(X_tr_tab_t[-n_val_tab:], X_tr_uns_tab_t[-n_val_tab:], y_tr_tab_t[-n_val_tab:]), batch_size=64)
    train_loader_hybrid_sub = DataLoader(TensorDataset(X_tr_tab_t[:-n_val_tab], X_tr_uns_tab_t[:-n_val_tab], y_tr_tab_t[:-n_val_tab]), batch_size=64, shuffle=True)
    
    # Note: Hybrid train function signature differs slightly, we need feature indices
    hybrid_feature_indices = {c: i for i, c in enumerate(X_train_tab.columns)}
    hybrid_model.train(train_loader_hybrid_sub, hybrid_feature_indices, val_loader=val_loader_hybrid, data_processor=proc_tabular, live_plot=False)
    
    print("\nEvaluating HYBRID Model on Test Set...")
    X_te_uns_tab_t = torch.tensor(X_test_uns_tab.values, dtype=torch.float32)
    test_loader_hybrid = DataLoader(TensorDataset(torch.tensor(X_test_tab.values, dtype=torch.float32), X_te_uns_tab_t, torch.tensor(y_test_tab.values, dtype=torch.float32).view(-1, 1)), batch_size=64)
    hybrid_model.evaluate_loader(test_loader_hybrid, hybrid_feature_indices, data_processor=proc_tabular, dataset_type="Test")
    """
    
    # 2. Evaluate PI-NODE Propeller Model
    print("\n" + "="*50)
    print("Training PI-NODE Propeller Model")
    print("="*50)
    feature_indices = {c: i for i, c in enumerate(X_train_t.columns)}
    
    # Identify Calm-Water vs Weather indices
    calm_water_columns = ['Speed-Through-Water', 'Fore draft_AMS', 'Aft draft_AMS', 'Trim_AMS', 'Propeller-Shaft-RPM']
    weather_columns = ['True-Wind-Speed', 'True-Wind-Direction', 'Wind_angle_BRG_WIND', 'Rel-Wind-Speed', 'Rel-Wind-Direction']
    
    calm_water_indices = [feature_indices[col] for col in calm_water_columns if col in feature_indices]
    weather_indices = [feature_indices[col] for col in weather_columns if col in feature_indices]
    
    all_known = set(calm_water_indices + weather_indices)
    for col, idx in feature_indices.items():
        if idx not in all_known:
            calm_water_indices.append(idx)
            
    print(f"Calm water features: {len(calm_water_indices)}, Weather features: {len(weather_indices)}")
    
    node_model = PINODEPropellerModel(
        input_size=X_tr_seq.shape[2],
        feature_indices=feature_indices,
        calm_water_indices=calm_water_indices,
        weather_indices=weather_indices,
        data_processor=proc_temp,
        hidden_size=64,
        ode_num_layers=2,
        lr=0.001,
        epochs=1000,
        batch_size=64,
        loss_function_choice='SmoothL1',
        encoder_mode='first',
    )
    
    n_val_seq = int(len(X_tr_seq) * 0.2)
    train_loader_seq = node_model.prepare_sequence_dataloader(
        X_tr_seq[:-n_val_seq], X_tr_uns_seq[:-n_val_seq], y_tr_seq[:-n_val_seq], shuffle=True
    )
    val_loader_seq = node_model.prepare_sequence_dataloader(
        X_tr_seq[-n_val_seq:], X_tr_uns_seq[-n_val_seq:], y_tr_seq[-n_val_seq:], shuffle=False
    )
    test_loader_seq = node_model.prepare_sequence_dataloader(
        X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False
    )
    
    node_model.train(train_loader_seq, val_loader=val_loader_seq, live_plot=False)
    
    print("\nEvaluating PI-NODE Propeller Model on Test Set...")
    _, node_rmse = node_model.evaluate_loader(test_loader_seq)
    print(f"PI-NODE Propeller TEST RMSE: {node_rmse:.2f} kW")

if __name__ == "__main__":
    main()
