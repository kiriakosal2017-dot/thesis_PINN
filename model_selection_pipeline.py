import argparse
import json
import time
import traceback
import gc
import torch
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


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _load_trials_from_results_csv(csv_path: Path, top_k: int = None, only_ok: bool = True) -> List[Trial]:
    df = pd.read_csv(csv_path)
    if only_ok and "status" in df.columns:
        df = df[df["status"] == "ok"]
    if "val_loss" in df.columns:
        df = df.sort_values("val_loss", ascending=True)
    if top_k is not None and top_k > 0:
        df = df.head(top_k)

    trials: List[Trial] = []
    for _, row in df.iterrows():
        params = json.loads(row["params_json"]) if isinstance(row.get("params_json"), str) else {}
        model_type = row.get("model_type", "LSTM_PINN")
        trials.append(Trial(model_type=model_type, params=params))
    return trials


def build_trials(model_types: List[str], lstm_grid: Dict = None) -> List[Trial]:
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
        effective_lstm_grid = {
            "hidden_size": [64, 128, 256],
            "num_layers": [1, 2],
            "dropout": [0.2, 0.3],
            "lr": [1e-3, 5e-4],
            "batch_size": [64, 128],
            "alpha": [1.0],
            "beta": [0.05, 0.1],
        }
        if lstm_grid is not None:
            for key, value in lstm_grid.items():
                if value:
                    effective_lstm_grid[key] = value

        for hidden_size, num_layers, dropout, lr, batch_size, alpha, beta in product(
            effective_lstm_grid["hidden_size"],
            effective_lstm_grid["num_layers"],
            effective_lstm_grid["dropout"],
            effective_lstm_grid["lr"],
            effective_lstm_grid["batch_size"],
            effective_lstm_grid["alpha"],
            effective_lstm_grid["beta"],
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


def _run_trial_list(
    trials: List[Trial],
    epochs: int,
    output_dir: Path,
    model_types: List[str],
    csv_filename: str,
    txt_filename: str,
    stage_label: str,
    tabular_data: Dict,
    temporal_data: Dict,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "trial_metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / csv_filename
    txt_path = output_dir / txt_filename

    rows = []
    for idx, trial in enumerate(trials, start=1):
        started_at = _now_iso()
        t0 = time.time()
        params_json = json.dumps(trial.params, sort_keys=True)
        model_size = _model_size_label(trial)
        ckpt_path = checkpoints_dir / f"{stage_label}_trial_{idx:04d}_{trial.model_type}.pt"
        trial_metrics_path = metrics_dir / f"{stage_label}_trial_{idx:04d}_{trial.model_type}.csv"

        print(
            f"\n[{idx}/{len(trials)}] Running {trial.model_type} | "
            f"size={model_size} | params={params_json}"
        )

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
        finally:
            # Force cleanup to prevent MPS Out-Of-Memory on Apple Silicon
            if "model" in locals():
                del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        duration_sec = time.time() - t0
        row = {
            "trial_id": idx,
            "stage": stage_label,
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

        stage_df = pd.DataFrame(rows)
        stage_df.to_csv(csv_path, index=False)
        _write_summary(stage_df, txt_path, epochs)

    return pd.DataFrame(rows)


def run_pipeline(
    epochs: int,
    max_trials: int,
    output_dir: Path,
    model_types: List[str],
    two_stage: bool = False,
    three_stage: bool = False,
    stage1_epochs: int = None,
    stage2_epochs: int = 100,
    stage2_top_k: int = 10,
    stage3_epochs: int = 250,
    stage3_top_k: int = 3,
    lstm_grid: Dict = None,
    resume_stage1_csv: Path = None,
    resume_stage2_csv: Path = None,
    retry_failed_stage1_csv: Path = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(model_types=model_types, lstm_grid=lstm_grid)
    if max_trials > 0:
        trials = trials[:max_trials]

    if resume_stage1_csv is not None or resume_stage2_csv is not None or retry_failed_stage1_csv is not None:
        print("Resume mode: ON")
        if resume_stage1_csv is not None:
            print(f"Using cached stage1 results from: {resume_stage1_csv}")
        if resume_stage2_csv is not None:
            print(f"Using cached stage2 results from: {resume_stage2_csv}")
        if retry_failed_stage1_csv is not None:
            print(f"Retrying failed stage1 trials from: {retry_failed_stage1_csv}")
    else:
        print(f"Total trials to run: {len(trials)}")
    if three_stage:
        two_stage = True

    if two_stage:
        effective_stage1 = stage1_epochs if stage1_epochs is not None else epochs
        if three_stage:
            print(
                "Three-stage mode: ON | "
                f"stage1_epochs={effective_stage1}, "
                f"stage2_epochs={stage2_epochs}, stage2_top_k={stage2_top_k}, "
                f"stage3_epochs={stage3_epochs}, stage3_top_k={stage3_top_k}"
            )
        else:
            print(f"Two-stage mode: ON | stage1_epochs={effective_stage1}, stage2_epochs={stage2_epochs}, top_k={stage2_top_k}")
    else:
        print(f"Epochs per trial: {epochs}")
    print(f"Output directory: {output_dir}")

    tabular_data = None
    temporal_data = None
    if any(mt in model_types for mt in ["DATA", "PGNN", "PINN"]):
        tabular_data = _prepare_tabular_data()
    if "LSTM_PINN" in model_types:
        temporal_data = _prepare_lstm_data()

    if not two_stage:
        stage_df = _run_trial_list(
            trials=trials,
            epochs=epochs,
            output_dir=output_dir,
            model_types=model_types,
            csv_filename="model_selection_trials.csv",
            txt_filename="model_selection_summary.txt",
            stage_label="single_stage",
            tabular_data=tabular_data,
            temporal_data=temporal_data,
        )
        print("\nPipeline completed.")
        print(f"Trials CSV: {output_dir / 'model_selection_trials.csv'}")
        print(f"Summary TXT: {output_dir / 'model_selection_summary.txt'}")
        return stage_df

    # Two-stage / Three-stage mode
    effective_stage1 = stage1_epochs if stage1_epochs is not None else epochs
    stage1_df = None
    stage2_trials: List[Trial] = []

    if resume_stage2_csv is not None:
        stage2_trials = _load_trials_from_results_csv(resume_stage2_csv, top_k=stage2_top_k, only_ok=False)
        print(f"\nStage2 resume from cached stage2 CSV with {len(stage2_trials)} trials.")
    elif retry_failed_stage1_csv is not None:
        cached_stage1_df = pd.read_csv(retry_failed_stage1_csv)
        if "status" in cached_stage1_df.columns:
            cached_ok_df = cached_stage1_df[cached_stage1_df["status"] == "ok"].copy()
            failed_df = cached_stage1_df[cached_stage1_df["status"] != "ok"].copy()
        else:
            cached_ok_df = cached_stage1_df.copy()
            failed_df = pd.DataFrame(columns=cached_stage1_df.columns)

        retry_trials: List[Trial] = []
        for _, row in failed_df.iterrows():
            params = json.loads(row["params_json"]) if isinstance(row.get("params_json"), str) else {}
            model_type = row.get("model_type", "LSTM_PINN")
            retry_trials.append(Trial(model_type=model_type, params=params))

        print(
            f"\nStage1 retry mode: cached_ok={len(cached_ok_df)}, "
            f"failed_to_retry={len(retry_trials)}"
        )
        if retry_trials:
            stage1_retry_df = _run_trial_list(
                trials=retry_trials,
                epochs=effective_stage1,
                output_dir=output_dir,
                model_types=model_types,
                csv_filename="stage1_retry_trials.csv",
                txt_filename="stage1_retry_summary.txt",
                stage_label="stage1_retry",
                tabular_data=tabular_data,
                temporal_data=temporal_data,
            )
            stage1_df = pd.concat([cached_ok_df, stage1_retry_df], ignore_index=True)
        else:
            stage1_df = cached_ok_df

        stage1_df.to_csv(output_dir / "stage1_trials.csv", index=False)
        _write_summary(stage1_df, output_dir / "stage1_summary.txt", effective_stage1)
        ok_stage1 = stage1_df[stage1_df["status"] == "ok"].copy().sort_values("val_loss", ascending=True)
        top_k = min(stage2_top_k, len(ok_stage1))
        stage2_trials = [
            Trial(
                model_type=row["model_type"],
                params=json.loads(row["params_json"]) if isinstance(row.get("params_json"), str) else {},
            )
            for _, row in ok_stage1.head(top_k).iterrows()
        ]
        print(f"Stage2 selection after retry: top-{len(stage2_trials)} trials.")
    elif resume_stage1_csv is not None:
        stage2_trials = _load_trials_from_results_csv(resume_stage1_csv, top_k=stage2_top_k, only_ok=True)
        print(f"\nStage2 resume from cached stage1 CSV with top-{len(stage2_trials)} trials.")
    else:
        stage1_df = _run_trial_list(
            trials=trials,
            epochs=effective_stage1,
            output_dir=output_dir,
            model_types=model_types,
            csv_filename="stage1_trials.csv",
            txt_filename="stage1_summary.txt",
            stage_label="stage1",
            tabular_data=tabular_data,
            temporal_data=temporal_data,
        )

        ok_stage1 = stage1_df[stage1_df["status"] == "ok"].copy().sort_values("val_loss", ascending=True)
        if ok_stage1.empty:
            print("\nStage1 produced no successful trials; skipping stage2.")
            stage1_df.to_csv(output_dir / "model_selection_trials.csv", index=False)
            _write_summary(stage1_df, output_dir / "model_selection_summary.txt", effective_stage1)
            return stage1_df

        top_k = min(stage2_top_k, len(ok_stage1))
        stage2_trials = [
            Trial(
                model_type=row["model_type"],
                params=json.loads(row["params_json"]) if isinstance(row.get("params_json"), str) else {},
            )
            for _, row in ok_stage1.head(top_k).iterrows()
        ]

    if not stage2_trials:
        print("\nNo stage2 trials available (resume source empty).")
        return pd.DataFrame()

    print(f"\nStage2 refinement on {len(stage2_trials)} trials.")
    stage2_df = _run_trial_list(
        trials=stage2_trials,
        epochs=stage2_epochs,
        output_dir=output_dir,
        model_types=model_types,
        csv_filename="stage2_trials.csv",
        txt_filename="stage2_summary.txt",
        stage_label="stage2",
        tabular_data=tabular_data,
        temporal_data=temporal_data,
    )

    if not three_stage:
        frames = [df for df in [stage1_df, stage2_df] if df is not None and len(df) > 0]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_csv(output_dir / "model_selection_trials.csv", index=False)
        _write_summary(stage2_df, output_dir / "model_selection_summary.txt", stage2_epochs)

        print("\nTwo-stage pipeline completed.")
        print(f"Stage1 CSV/TXT: {output_dir / 'stage1_trials.csv'}, {output_dir / 'stage1_summary.txt'}")
        print(f"Stage2 CSV/TXT: {output_dir / 'stage2_trials.csv'}, {output_dir / 'stage2_summary.txt'}")
        print(f"Combined CSV: {output_dir / 'model_selection_trials.csv'}")
        print(f"Final summary TXT: {output_dir / 'model_selection_summary.txt'}")
        return combined

    ok_stage2 = stage2_df[stage2_df["status"] == "ok"].copy().sort_values("val_loss", ascending=True)
    if ok_stage2.empty:
        print("\nStage2 produced no successful trials; skipping stage3.")
        frames = [df for df in [stage1_df, stage2_df] if df is not None and len(df) > 0]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_csv(output_dir / "model_selection_trials.csv", index=False)
        _write_summary(stage2_df, output_dir / "model_selection_summary.txt", stage2_epochs)
        return combined

    top_k3 = min(stage3_top_k, len(ok_stage2))
    top_stage2_ids = ok_stage2.head(top_k3)["trial_id"].tolist()
    stage3_trials = [stage2_trials[trial_id - 1] for trial_id in top_stage2_ids]

    print(f"\nStage3 confirmation on top-{top_k3} trials from stage2.")
    stage3_df = _run_trial_list(
        trials=stage3_trials,
        epochs=stage3_epochs,
        output_dir=output_dir,
        model_types=model_types,
        csv_filename="stage3_trials.csv",
        txt_filename="stage3_summary.txt",
        stage_label="stage3",
        tabular_data=tabular_data,
        temporal_data=temporal_data,
    )

    frames = [df for df in [stage1_df, stage2_df, stage3_df] if df is not None and len(df) > 0]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(output_dir / "model_selection_trials.csv", index=False)
    _write_summary(stage3_df, output_dir / "model_selection_summary.txt", stage3_epochs)

    print("\nThree-stage pipeline completed.")
    print(f"Stage1 CSV/TXT: {output_dir / 'stage1_trials.csv'}, {output_dir / 'stage1_summary.txt'}")
    print(f"Stage2 CSV/TXT: {output_dir / 'stage2_trials.csv'}, {output_dir / 'stage2_summary.txt'}")
    print(f"Stage3 CSV/TXT: {output_dir / 'stage3_trials.csv'}, {output_dir / 'stage3_summary.txt'}")
    print(f"Combined CSV: {output_dir / 'model_selection_trials.csv'}")
    print(f"Final summary TXT: {output_dir / 'model_selection_summary.txt'}")
    return combined


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
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Enable two-stage selection: coarse sweep then top-k refinement.",
    )
    parser.add_argument(
        "--three-stage",
        action="store_true",
        help="Enable three-stage selection: coarse sweep, refine top-k, then confirm top-k.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="Epochs for stage1 (defaults to --epochs).",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=100,
        help="Epochs for stage2 top-k refinement.",
    )
    parser.add_argument(
        "--stage2-top-k",
        type=int,
        default=10,
        help="How many top stage1 trials to re-run in stage2.",
    )
    parser.add_argument(
        "--stage3-epochs",
        type=int,
        default=250,
        help="Epochs for stage3 top-k confirmation.",
    )
    parser.add_argument(
        "--stage3-top-k",
        type=int,
        default=3,
        help="How many top stage2 trials to re-run in stage3.",
    )
    parser.add_argument(
        "--resume-stage1-csv",
        type=str,
        default=None,
        help="Path to an existing stage1_trials.csv to skip stage1 and run stage2/stage3 directly.",
    )
    parser.add_argument(
        "--resume-stage2-csv",
        type=str,
        default=None,
        help="Path to an existing stage2_trials.csv to rerun stage2/stage3 directly.",
    )
    parser.add_argument(
        "--retry-failed-stage1-csv",
        type=str,
        default=None,
        help="Path to an existing stage1_trials.csv; rerun only failed stage1 trials, then continue with stage2/stage3.",
    )
    parser.add_argument(
        "--lstm-hidden-sizes",
        type=str,
        default="64,128,256",
        help="Comma-separated hidden sizes for LSTM grid.",
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=str,
        default="1,2",
        help="Comma-separated number of LSTM layers for grid.",
    )
    parser.add_argument(
        "--lstm-dropouts",
        type=str,
        default="0.2,0.3",
        help="Comma-separated LSTM dropout values for grid.",
    )
    parser.add_argument(
        "--lstm-lrs",
        type=str,
        default="0.001,0.0005",
        help="Comma-separated learning rates for LSTM grid.",
    )
    parser.add_argument(
        "--lstm-batch-sizes",
        type=str,
        default="64,128",
        help="Comma-separated batch sizes for LSTM grid.",
    )
    parser.add_argument(
        "--lstm-betas",
        type=str,
        default="0.05,0.1",
        help="Comma-separated physics-loss beta values for LSTM grid.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    allowed = {"DATA", "PGNN", "PINN", "LSTM_PINN"}
    invalid = [m for m in model_types if m not in allowed]
    if invalid:
        raise ValueError(f"Unknown model type(s): {invalid}. Allowed: {sorted(allowed)}")

    lstm_grid = {
        "hidden_size": _parse_int_list(args.lstm_hidden_sizes),
        "num_layers": _parse_int_list(args.lstm_num_layers),
        "dropout": _parse_float_list(args.lstm_dropouts),
        "lr": _parse_float_list(args.lstm_lrs),
        "batch_size": _parse_int_list(args.lstm_batch_sizes),
        "alpha": [1.0],
        "beta": _parse_float_list(args.lstm_betas),
    }

    run_pipeline(
        epochs=args.epochs,
        max_trials=args.max_trials,
        output_dir=Path(args.output_dir),
        model_types=model_types,
        two_stage=args.two_stage,
        three_stage=args.three_stage,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_top_k=args.stage2_top_k,
        stage3_epochs=args.stage3_epochs,
        stage3_top_k=args.stage3_top_k,
        lstm_grid=lstm_grid,
        resume_stage1_csv=Path(args.resume_stage1_csv) if args.resume_stage1_csv else None,
        resume_stage2_csv=Path(args.resume_stage2_csv) if args.resume_stage2_csv else None,
        retry_failed_stage1_csv=Path(args.retry_failed_stage1_csv) if args.retry_failed_stage1_csv else None,
    )
