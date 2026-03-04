import argparse
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DataConfig, TrainingConfig
from read_data import DataProcessor, create_sequences
from main_DATA import DataDrivenModel
from main_PGNN import PGNNModel, _build_feature_indices as build_pgnn_feature_indices
from main_PINN import PINNModel, _build_feature_indices as build_pinn_feature_indices
from main_LSTM import LSTMPINNModel, _build_feature_indices as build_lstm_feature_indices


@dataclass
class Trial:
    model_type: str
    params: Dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_trials(model_types: List[str]) -> List[Trial]:
    trials: List[Trial] = []

    if "DATA" in model_types:
        data_grid = {
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128],
        }
        for lr, batch_size in product(data_grid["lr"], data_grid["batch_size"]):
            trials.append(Trial("DATA", {"lr": lr, "batch_size": batch_size}))

    if "PGNN" in model_types:
        pgnn_grid = {
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128],
            "alpha": [1.0],
            "beta": [0.05, 0.1],
            "k_wave": [1e-7, 1e-6],
        }
        for lr, batch_size, alpha, beta, k_wave in product(
            pgnn_grid["lr"],
            pgnn_grid["batch_size"],
            pgnn_grid["alpha"],
            pgnn_grid["beta"],
            pgnn_grid["k_wave"],
        ):
            trials.append(
                Trial(
                    "PGNN",
                    {
                        "lr": lr,
                        "batch_size": batch_size,
                        "alpha": alpha,
                        "beta": beta,
                        "k_wave": k_wave,
                    },
                )
            )

    if "PINN" in model_types:
        pinn_grid = {
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128],
            "alpha": [1.0],
            "beta": [0.05, 0.1],
            "gamma": [0.05, 0.1],
        }
        for lr, batch_size, alpha, beta, gamma in product(
            pinn_grid["lr"],
            pinn_grid["batch_size"],
            pinn_grid["alpha"],
            pinn_grid["beta"],
            pinn_grid["gamma"],
        ):
            trials.append(
                Trial(
                    "PINN",
                    {
                        "lr": lr,
                        "batch_size": batch_size,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                    },
                )
            )

    if "LSTM_PINN" in model_types:
        lstm_grid = {
            "hidden_size": [64, 128, 256],
            "num_layers": [1, 2],
            "dropout": [0.2, 0.3],
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128],
            "alpha": [1.0],
            "beta": [0.05, 0.1],
        }
        for hidden_size, num_layers, dropout, lr, batch_size, alpha, beta in product(
            lstm_grid["hidden_size"],
            lstm_grid["num_layers"],
            lstm_grid["dropout"],
            lstm_grid["lr"],
            lstm_grid["batch_size"],
            lstm_grid["alpha"],
            lstm_grid["beta"],
        ):
            trials.append(
                Trial(
                    "LSTM_PINN",
                    {
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "lr": lr,
                        "batch_size": batch_size,
                        "alpha": alpha,
                        "beta": beta,
                    },
                )
            )

    return trials


def _model_size_label(trial: Trial) -> str:
    if trial.model_type == "LSTM_PINN":
        return (
            f"hs{trial.params['hidden_size']}_"
            f"nl{trial.params['num_layers']}_"
            f"do{trial.params['dropout']}"
        )
    return "mlp_default"


def _prepare_tabular_data() -> Dict:
    processor = DataProcessor()
    result = processor.load_and_prepare_data()
    if result is None:
        raise RuntimeError("Failed to load tabular data")

    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = result
    return {
        "processor": processor,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_uns": X_train_uns,
        "X_test_uns": X_test_uns,
        "y_train": y_train,
        "y_test": y_test,
    }


def _prepare_lstm_data(seq_len: int = 10) -> Dict:
    processor = DataProcessor()
    result = processor.load_and_prepare_temporal_data()
    if result is None:
        raise RuntimeError("Failed to load temporal data")

    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = result
    X_train_seq, X_train_uns_seq, y_train_seq = create_sequences(
        X_train, X_train_uns, y_train, seq_length=seq_len
    )
    X_test_seq, X_test_uns_seq, y_test_seq = create_sequences(
        X_test, X_test_uns, y_test, seq_length=seq_len
    )

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        raise RuntimeError("No valid LSTM sequences created")

    feature_indices = build_lstm_feature_indices(X_train.columns)
    return {
        "processor": processor,
        "feature_indices": feature_indices,
        "X_train_seq": X_train_seq,
        "X_train_uns_seq": X_train_uns_seq,
        "y_train_seq": y_train_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
    }


def _run_data_trial(trial: Trial, tabular: Dict, epochs: int, ckpt_path: Path) -> Tuple[float, float]:
    X_train, X_val, y_train, y_val = train_test_split(
        tabular["X_train"],
        tabular["y_train"],
        test_size=DataConfig.TEST_SIZE,
        random_state=DataConfig.RANDOM_STATE,
    )

    model = DataDrivenModel(
        input_size=tabular["X_train"].shape[1],
        lr=trial.params["lr"],
        epochs=epochs,
        batch_size=trial.params["batch_size"],
        optimizer_choice=TrainingConfig.OPTIMIZER,
        loss_function_choice=TrainingConfig.LOSS_FUNCTION,
    )
    train_loader = model.prepare_dataloader(X_train, y_train)
    val_loader = model.prepare_dataloader(X_val, y_val)

    model.train(train_loader, val_loader=val_loader, live_plot=False, checkpoint_path=str(ckpt_path))
    val_loss = model.evaluate(X_val, y_val, dataset_type="Validation", data_processor=tabular["processor"])
    test_loss = model.evaluate(
        tabular["X_test"], tabular["y_test"], dataset_type="Test", data_processor=tabular["processor"]
    )
    return val_loss, test_loss


def _run_pgnn_trial(trial: Trial, tabular: Dict, epochs: int, ckpt_path: Path) -> Tuple[float, float]:
    X_train, X_val, X_train_uns, X_val_uns, y_train, y_val = train_test_split(
        tabular["X_train"],
        tabular["X_train_uns"],
        tabular["y_train"],
        test_size=DataConfig.TEST_SIZE,
        random_state=DataConfig.RANDOM_STATE,
    )
    feature_indices = build_pgnn_feature_indices(X_train_uns)

    model = PGNNModel(
        input_size=tabular["X_train"].shape[1],
        lr=trial.params["lr"],
        epochs=epochs,
        batch_size=trial.params["batch_size"],
        optimizer_choice=TrainingConfig.OPTIMIZER,
        loss_function_choice=TrainingConfig.LOSS_FUNCTION,
        alpha=trial.params["alpha"],
        beta=trial.params["beta"],
        k_wave=trial.params["k_wave"],
    )
    train_loader = model.prepare_dataloader(X_train, y_train)
    unscaled_loader = model.prepare_unscaled_dataloader(X_train_uns)
    val_loader = model.prepare_dataloader(X_val, y_val)

    model.train(
        train_loader,
        unscaled_loader,
        feature_indices,
        tabular["processor"],
        val_loader=val_loader,
        live_plot=False,
        checkpoint_path=str(ckpt_path),
    )
    val_loss = model.evaluate(X_val, y_val, dataset_type="Validation", data_processor=tabular["processor"])
    test_loss = model.evaluate(
        tabular["X_test"], tabular["y_test"], dataset_type="Test", data_processor=tabular["processor"]
    )
    return val_loss, test_loss


def _run_pinn_trial(trial: Trial, tabular: Dict, epochs: int, ckpt_path: Path) -> Tuple[float, float]:
    X_train, X_val, X_train_uns, X_val_uns, y_train, y_val = train_test_split(
        tabular["X_train"],
        tabular["X_train_uns"],
        tabular["y_train"],
        test_size=DataConfig.TEST_SIZE,
        random_state=DataConfig.RANDOM_STATE,
    )
    feature_indices = build_pinn_feature_indices(X_train_uns)

    model = PINNModel(
        input_size=tabular["X_train"].shape[1],
        lr=trial.params["lr"],
        epochs=epochs,
        batch_size=trial.params["batch_size"],
        optimizer_choice=TrainingConfig.OPTIMIZER,
        loss_function_choice=TrainingConfig.LOSS_FUNCTION,
        alpha=trial.params["alpha"],
        beta=trial.params["beta"],
        gamma=trial.params["gamma"],
    )
    train_loader = model.prepare_dataloader(X_train, y_train)
    val_loader = model.prepare_dataloader(X_val, y_val)

    model.train(
        train_loader,
        X_train_uns,
        feature_indices,
        tabular["processor"],
        val_loader=val_loader,
        live_plot=False,
        checkpoint_path=str(ckpt_path),
    )
    val_loss = model.evaluate(X_val, y_val, dataset_type="Validation", data_processor=tabular["processor"])
    test_loss = model.evaluate(
        tabular["X_test"], tabular["y_test"], dataset_type="Test", data_processor=tabular["processor"]
    )
    return val_loss, test_loss


def _run_lstm_trial(
    trial: Trial, temporal: Dict, epochs: int, ckpt_path: Path, trial_metrics_path: Path
) -> Tuple[float, float]:
    X_train_seq = temporal["X_train_seq"]
    X_train_uns_seq = temporal["X_train_uns_seq"]
    y_train_seq = temporal["y_train_seq"]

    n_val = max(1, int(len(X_train_seq) * DataConfig.TEST_SIZE))
    X_final_train = X_train_seq[:-n_val]
    X_final_uns_train = X_train_uns_seq[:-n_val]
    y_final_train = y_train_seq[:-n_val]

    X_final_val = X_train_seq[-n_val:]
    X_final_uns_val = X_train_uns_seq[-n_val:]
    y_final_val = y_train_seq[-n_val:]

    model = LSTMPINNModel(
        input_size=X_train_seq.shape[2],
        hidden_size=trial.params["hidden_size"],
        num_layers=trial.params["num_layers"],
        dropout=trial.params["dropout"],
        lr=trial.params["lr"],
        epochs=epochs,
        batch_size=trial.params["batch_size"],
        optimizer_choice=TrainingConfig.OPTIMIZER,
        loss_function_choice=TrainingConfig.LOSS_FUNCTION,
        alpha=trial.params["alpha"],
        beta=trial.params["beta"],
        weight_decay=TrainingConfig.WEIGHT_DECAY,
    )
    train_loader = model.prepare_sequence_dataloader(X_final_train, X_final_uns_train, y_final_train)
    val_loader = model.prepare_sequence_dataloader(X_final_val, X_final_uns_val, y_final_val)

    model.train(
        train_loader,
        temporal["feature_indices"],
        temporal["processor"],
        val_loader=val_loader,
        live_plot=False,
        metrics_output_path=str(trial_metrics_path),
        checkpoint_path=str(ckpt_path),
    )
    val_loss = model.evaluate(
        X_final_val, y_final_val, dataset_type="Validation", data_processor=temporal["processor"]
    )
    test_loss = model.evaluate(
        temporal["X_test_seq"],
        temporal["y_test_seq"],
        dataset_type="Test",
        data_processor=temporal["processor"],
    )
    return val_loss, test_loss


def _write_summary(results_df: pd.DataFrame, summary_path: Path, epochs: int):
    lines: List[str] = []
    lines.append("Model Selection Summary")
    lines.append("=======================")
    lines.append(f"Generated at: {_now_iso()}")
    lines.append(f"Epochs per trial: {epochs}")
    lines.append(f"Successful trials: {(results_df['status'] == 'ok').sum()} / {len(results_df)}")
    lines.append("")

    ok_df = results_df[results_df["status"] == "ok"].copy()
    if ok_df.empty:
        lines.append("No successful trials.")
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        return

    ok_df = ok_df.sort_values("val_loss", ascending=True)
    best_overall = ok_df.iloc[0]
    lines.append("Best Overall Trial")
    lines.append("------------------")
    lines.append(f"Model: {best_overall['model_type']}")
    lines.append(f"Size: {best_overall['model_size']}")
    lines.append(f"Validation loss: {best_overall['val_loss']:.8f}")
    lines.append(f"Test loss: {best_overall['test_loss']:.8f}")
    lines.append(f"Params: {best_overall['params_json']}")
    lines.append("")

    lines.append("Best Per Model Type")
    lines.append("-------------------")
    for model_type, group in ok_df.groupby("model_type"):
        row = group.sort_values("val_loss", ascending=True).iloc[0]
        lines.append(
            f"- {model_type}: val={row['val_loss']:.8f}, "
            f"test={row['test_loss']:.8f}, size={row['model_size']}, params={row['params_json']}"
        )
    lines.append("")

    lines.append("Top 10 Trials by Validation Loss")
    lines.append("--------------------------------")
    for _, row in ok_df.head(10).iterrows():
        lines.append(
            f"- #{int(row['trial_id'])} {row['model_type']} [{row['model_size']}] "
            f"val={row['val_loss']:.8f} test={row['test_loss']:.8f}"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(epochs: int, max_trials: int, output_dir: Path, model_types: List[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "trial_metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "model_selection_trials.csv"
    txt_path = output_dir / "model_selection_summary.txt"

    trials = build_trials(model_types=model_types)
    if max_trials > 0:
        trials = trials[:max_trials]

    print(f"Total trials to run: {len(trials)}")
    print(f"Epochs per trial: {epochs}")
    print(f"Output directory: {output_dir}")

    tabular_data = None
    temporal_data = None
    if any(mt in model_types for mt in ["DATA", "PGNN", "PINN"]):
        tabular_data = _prepare_tabular_data()
    if "LSTM_PINN" in model_types:
        temporal_data = _prepare_lstm_data()

    rows = []
    for idx, trial in enumerate(trials, start=1):
        started_at = _now_iso()
        t0 = time.time()
        params_json = json.dumps(trial.params, sort_keys=True)
        model_size = _model_size_label(trial)
        ckpt_path = checkpoints_dir / f"trial_{idx:04d}_{trial.model_type}.pt"
        trial_metrics_path = metrics_dir / f"trial_{idx:04d}_{trial.model_type}.csv"

        print(f"\n[{idx}/{len(trials)}] Running {trial.model_type} | size={model_size} | params={params_json}")

        status = "ok"
        error_message = ""
        val_loss = None
        test_loss = None
        try:
            if trial.model_type == "DATA":
                val_loss, test_loss = _run_data_trial(trial, tabular_data, epochs, ckpt_path)
            elif trial.model_type == "PGNN":
                val_loss, test_loss = _run_pgnn_trial(trial, tabular_data, epochs, ckpt_path)
            elif trial.model_type == "PINN":
                val_loss, test_loss = _run_pinn_trial(trial, tabular_data, epochs, ckpt_path)
            elif trial.model_type == "LSTM_PINN":
                val_loss, test_loss = _run_lstm_trial(
                    trial, temporal_data, epochs, ckpt_path, trial_metrics_path
                )
            else:
                raise ValueError(f"Unknown model type: {trial.model_type}")
        except Exception as exc:
            status = "failed"
            error_message = f"{exc.__class__.__name__}: {exc}"
            print(f"Trial failed: {error_message}")
            print(traceback.format_exc())

        duration_sec = time.time() - t0
        row = {
            "trial_id": idx,
            "started_at": started_at,
            "ended_at": _now_iso(),
            "duration_sec": round(duration_sec, 3),
            "model_type": trial.model_type,
            "model_size": model_size,
            "params_json": params_json,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "status": status,
            "error": error_message,
        }
        rows.append(row)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        _write_summary(pd.DataFrame(rows), txt_path, epochs)

    print("\nPipeline completed.")
    print(f"Trials CSV: {csv_path}")
    print(f"Summary TXT: {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model-type and hyperparameter selection pipeline with CSV/TXT outputs."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingConfig.EPOCHS_CV,
        help="Epochs per trial.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="If >0, run only the first N trials (useful for smoke testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_selection",
        help="Directory for CSV/TXT results and trial artifacts.",
    )
    parser.add_argument(
        "--model-types",
        type=str,
        default="DATA,PGNN,PINN,LSTM_PINN",
        help="Comma-separated list of models to evaluate: DATA,PGNN,PINN,LSTM_PINN",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    allowed = {"DATA", "PGNN", "PINN", "LSTM_PINN"}
    invalid = [m for m in model_types if m not in allowed]
    if invalid:
        raise ValueError(f"Unknown model type(s): {invalid}. Allowed: {sorted(allowed)}")

    run_pipeline(
        epochs=args.epochs,
        max_trials=args.max_trials,
        output_dir=Path(args.output_dir),
        model_types=model_types,
    )
