"""Train the final PI-NODE model on the DANAE dataset using the winning regularization
hyperparameters from the sweep, then persist the checkpoint and fitted data processor
so downstream transfer-learning scripts can consume them directly.
"""
import pickle
import torch
from read_data import DataProcessor, create_sequences, split_calm_weather_indices
from config import DataConfig, SequenceConfig
from main_PI_NODE_Propeller import PINODEPropellerModel


def main():
    # RPM is dropped by default in DataConfig for some pipelines; the PI-NODE needs it
    # as a physics input, so restore it before loading.
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    # Load and scale the full temporal dataset (DANAE); the processor is serialised
    # at the end so that transfer scripts share identical scaler state.
    print("Loading temporal data (DANAE)...")
    proc = DataProcessor()
    res = proc.load_and_prepare_temporal_data()
    if res is None:
        raise RuntimeError("Failed to load temporal data.")

    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = res

    # Column-to-index mapping lets the physics loss address specific features by name
    # without hard-coding positional constants throughout the model.
    feature_indices = {c: i for i, c in enumerate(X_train.columns)}

    # Calm-water vs. weather feature split drives the structured physics residuals
    # inside the ODE right-hand side.
    calm_water_indices, weather_indices = split_calm_weather_indices(X_train.columns)
    print(f"Calm-water features: {len(calm_water_indices)}, Weather features: {len(weather_indices)}")

    # Wrap raw tabular data into overlapping windows; the ODE integrator consumes
    # the full sequence, not just the last time step.
    seq_len = SequenceConfig.LENGTH
    X_tr_seq, X_tr_uns_seq, y_tr_seq = create_sequences(X_train, X_train_uns, y_train, seq_length=seq_len)
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(X_test, X_test_uns, y_test, seq_length=seq_len)

    # Hold out the most-recent 20% of training sequences as a chronological validation
    # set; random shuffling would leak future context into the past.
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

    # Winning regularization weights selected via sweep: enforce monotone KQ-range,
    # penalise curvature, and pull KQ toward the Wageningen B-series prior.
    model.LAMBDA_KQ_RANGE = 0.25
    model.LAMBDA_KQ_CURVATURE = 0.01
    model.LAMBDA_KQ_PRIOR = 0.001

    # Chronological slicing: training sequences come first, validation from the tail.
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
    print("Final PI-NODE training (selected regularisation config)")
    print("  lambda_range=0.25, lambda_curv=0.01, lambda_prior=0.001")
    print("=" * 60)

    # Best validation checkpoint written to disk; training history logged for
    # post-hoc learning-curve analysis.
    model.train(
        train_loader,
        val_loader=val_loader,
        live_plot=False,
        checkpoint_path="best_model_PI_NODE_danae.pt",
        history_csv="results/history/PI_NODE_danae.csv",
    )

    print("\nEvaluating on Test Set...")
    _, test_rmse = model.evaluate_loader(test_loader)
    print(f"\nFINAL PI-NODE TEST RMSE: {test_rmse:.2f} kW")

    # Persist the fitted DataProcessor alongside the checkpoint so that any script
    # loading the weights can invert the same scaler without reprocessing raw data.
    with open("data_processor_danae_temporal.pkl", "wb") as f:
        pickle.dump(proc, f)
    print("Saved best_model_PI_NODE_danae.pt and data_processor_danae_temporal.pkl")

    print("\n" + "=" * 60)
    print("Training complete -- summary:")
    print(f"  DATA    Test RMSE:  424.80 kW")
    print(f"  HYBRID  Test RMSE:  703.69 kW")
    print(f"  PI-NODE Test RMSE:  {test_rmse:.2f} kW")
    print("=" * 60)


if __name__ == "__main__":
    main()
