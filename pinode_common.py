"""Shared helpers for PI-NODE ablation / multi-seed / uncertainty experiments.

Centralises the source vessel temporal data loading, sequence building, calm/weather
feature split, loader construction and metric computation so the experiment
scripts stay short and consistent with the main training pipeline.
"""
import numpy as np
import torch

from read_data import DataProcessor, create_sequences, split_calm_weather_indices
from config import DataConfig, SequenceConfig


def load_danae_temporal_sequences(meta_exclude=("dt", "acceleration")):
    """Load source vessel temporal data and build train/test sequences.

    Returns: (proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple)
    where each *_tuple is (X_seq, X_unscaled_seq, y_seq).

    ``meta_exclude`` controls which derived columns are kept out of the model
    branches. Pass ``('dt',)`` to deliberately include ``acceleration`` as a
    feature (the "with_acceleration" ablation).
    """
    # Propeller-Shaft-RPM is dropped globally in DataConfig for tabular models
    # but the PI-NODE physics branch requires it; ensure it is present before
    # loading source vessel temporal data.
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    proc = DataProcessor()
    res = proc.load_and_prepare_temporal_data()
    if res is None:
        raise RuntimeError("Failed to load source vessel temporal data")
    # Unpack the eight-element tuple; the last two elements (scalers) are
    # accessed through proc directly and not needed here.
    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = res

    # Build a column-name -> index map used by the PI-NODE to address specific
    # physics inputs (RPM, speed, etc.) without hardcoding column positions.
    feature_indices = {c: i for i, c in enumerate(X_train.columns)}
    calm_idx, weather_idx = split_calm_weather_indices(X_train.columns, exclude=meta_exclude)

    seq_len = SequenceConfig.LENGTH
    train_tuple = create_sequences(X_train, X_train_uns, y_train, seq_length=seq_len)
    test_tuple = create_sequences(X_test, X_test_uns, y_test, seq_length=seq_len)
    return proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple


def make_loaders(model, train_tuple, test_tuple, val_frac=0.2):
    """Build chronological train/val/test sequence loaders for a PINODE model."""
    X_tr_seq, X_tr_uns_seq, y_tr_seq = train_tuple
    X_te_seq, X_te_uns_seq, y_te_seq = test_tuple

    # Reserve the last val_frac of the training window as validation; taking
    # the tail preserves temporal ordering and avoids leaking future data.
    n_val = int(len(X_tr_seq) * val_frac)

    train_loader = model.prepare_sequence_dataloader(
        X_tr_seq[:-n_val], X_tr_uns_seq[:-n_val], y_tr_seq[:-n_val], shuffle=True
    )
    val_loader = model.prepare_sequence_dataloader(
        X_tr_seq[-n_val:], X_tr_uns_seq[-n_val:], y_tr_seq[-n_val:], shuffle=False
    )
    # Test set is never shuffled to allow aligned comparison of predicted vs.
    # true power traces along the voyage timeline.
    test_loader = model.prepare_sequence_dataloader(
        X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False
    )
    return train_loader, val_loader, test_loader


def predict_power(model, loader, mc_dropout=False):
    """Return (preds_kW, true_kW) arrays for a loader.

    If ``mc_dropout`` is True, dropout layers are kept active (stochastic forward
    pass) for Monte-Carlo Dropout uncertainty estimation.
    """
    # Start in eval mode to disable batch-norm running-stat updates; dropout
    # is selectively re-enabled below only for the MC-Dropout UQ path.
    model.model.eval()
    if mc_dropout:
        # Re-enable dropout while keeping all other layers in eval mode so that
        # repeated forward passes sample from the approximate posterior.
        for m in model.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    preds, true = [], []
    with torch.no_grad():
        for X_b, X_uns_b, y_b in loader:
            X_b = X_b.to(model.device)
            X_uns_b = X_uns_b.to(model.device)
            # The PI-NODE head returns angular velocity (w), thrust (t) and
            # rotative efficiency (eta_r); analytical power is derived from these.
            w, t, eta_r = model.model(X_b)
            # Use only the last time-step of the unscaled sequence as the physics
            # inputs because shaft-power is predicted for that instant.
            P_kw, _, _, _, _ = model.compute_analytical_power(w, t, eta_r, X_uns_b[:, -1, :])
            preds.append(P_kw.cpu().numpy())
            # Inverse-transform labels to kW to match the unit of P_kw.
            true.append(model.data_processor.inverse_transform_y(y_b.numpy()))

    preds = np.concatenate(preds).reshape(-1)
    true = np.concatenate(true).reshape(-1)
    return preds, true


def rmse_mape(preds, true):
    """Compute RMSE (kW) and MAPE (%). MAPE denominator is floored at 100 kW."""
    rmse = float(np.sqrt(np.mean((preds - true) ** 2)))
    # Floor the denominator at 100 kW to prevent near-zero true-power samples
    # (manoeuvring / drift conditions) from inflating MAPE to unrealistic levels.
    mape = float(np.mean(np.abs((preds - true) / np.maximum(np.abs(true), 100.0))) * 100)
    return rmse, mape
