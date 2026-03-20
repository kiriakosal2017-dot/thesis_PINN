"""Phase 2.2: Train the final PI-NODE model with the winning regularization config
and save weights + data processor for Transfer Learning (Phases 4-5).

Winning config from sweep: lambda_range=0.25, lambda_curv=0.01, lambda_prior=0.001
"""
import pickle
import torch
from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig
from main_PI_NODE_Propeller import PINODEPropellerModel


def main():
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    print("Loading temporal data (DANAE)...")
    proc = DataProcessor()
    res = proc.load_and_prepare_temporal_data()
    if res is None:
        raise RuntimeError("Failed to load temporal data.")

    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = res
    feature_indices = {c: i for i, c in enumerate(X_train.columns)}

    calm_water_cols = [
        col for col in X_train.columns
        if not any(w in col.lower() for w in ['wind', 'wave', 'swell'])
    ]
    weather_cols = [
        col for col in X_train.columns
        if any(w in col.lower() for w in ['wind', 'wave', 'swell'])
    ]
    calm_water_indices = [feature_indices[col] for col in calm_water_cols]
    weather_indices = [feature_indices[col] for col in weather_cols]
    print(f"Calm-water features: {len(calm_water_indices)}, Weather features: {len(weather_indices)}")

    seq_len = SequenceConfig.LENGTH
    X_tr_seq, X_tr_uns_seq, y_tr_seq = create_sequences(X_train, X_train_uns, y_train, seq_length=seq_len)
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(X_test, X_test_uns, y_test, seq_length=seq_len)

    n_val = int(len(X_tr_seq) * 0.2)

    model = PINODEPropellerModel(
        input_size=X_tr_seq.shape[2],
        feature_indices=feature_indices,
        calm_water_indices=calm_water_indices,
        weather_indices=weather_indices,
        data_processor=proc,
        hidden_size=64,
        ode_num_layers=2,
        lr=0.001,
        epochs=1000,
        batch_size=64,
        loss_function_choice="SmoothL1",
        encoder_mode="first",
    )

    # Winning regularization weights from sweep
    model.LAMBDA_KQ_RANGE = 0.25
    model.LAMBDA_KQ_CURVATURE = 0.01
    model.LAMBDA_KQ_PRIOR = 0.001

    train_loader = model.prepare_sequence_dataloader(
        X_tr_seq[:-n_val], X_tr_uns_seq[:-n_val], y_tr_seq[:-n_val], shuffle=True
    )
    val_loader = model.prepare_sequence_dataloader(
        X_tr_seq[-n_val:], X_tr_uns_seq[-n_val:], y_tr_seq[-n_val:], shuffle=False
    )
    test_loader = model.prepare_sequence_dataloader(
        X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False
    )

    print("\n" + "=" * 60)
    print("PHASE 2.2: Final PI-NODE Training (Winning Config)")
    print("  lambda_range=0.25, lambda_curv=0.01, lambda_prior=0.001")
    print("=" * 60)

    model.train(
        train_loader,
        val_loader=val_loader,
        live_plot=False,
        checkpoint_path="best_model_PI_NODE_danae.pt",
    )

    print("\nEvaluating on Test Set...")
    _, test_rmse = model.evaluate_loader(test_loader)
    print(f"\nFINAL PI-NODE TEST RMSE: {test_rmse:.2f} kW")

    with open("data_processor_danae_temporal.pkl", "wb") as f:
        pickle.dump(proc, f)
    print("Saved best_model_PI_NODE_danae.pt and data_processor_danae_temporal.pkl")

    print("\n" + "=" * 60)
    print("PHASE 1+2 COMPLETE -- Summary:")
    print(f"  DATA    Test RMSE:  424.80 kW")
    print(f"  HYBRID  Test RMSE:  703.69 kW")
    print(f"  PI-NODE Test RMSE:  {test_rmse:.2f} kW")
    print("=" * 60)


if __name__ == "__main__":
    main()
