import csv
from pathlib import Path

from read_data import DataProcessor, create_sequences
from config import DataConfig, SequenceConfig
from main_PI_NODE_Propeller import PINODEPropellerModel


def main():
    # Ensure RPM feature is available for propeller physics.
    if "Propeller-Shaft-RPM" in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove("Propeller-Shaft-RPM")

    print("Loading temporal data...")
    proc = DataProcessor()
    res = proc.load_and_prepare_temporal_data()
    if res is None:
        raise RuntimeError("Failed to load temporal data.")

    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, _, _ = res
    feature_indices = {c: i for i, c in enumerate(X_train.columns)}

    # Identify calm-water and weather indices based on column names
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

    seq_len = SequenceConfig.LENGTH
    X_tr_seq, X_tr_uns_seq, y_tr_seq = create_sequences(
        X_train, X_train_uns, y_train, seq_length=seq_len
    )
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(
        X_test, X_test_uns, y_test, seq_length=seq_len
    )

    n_val = int(len(X_tr_seq) * 0.2)

    # Small, focused sweep around your current strong setup.
    sweep = [
        {"name": "baseline", "range": 0.50, "curv": 0.0100, "prior": 0.0010},
        {"name": "range_low", "range": 0.25, "curv": 0.0100, "prior": 0.0010},
        {"name": "range_high", "range": 0.75, "curv": 0.0100, "prior": 0.0010},
        {"name": "curv_low", "range": 0.50, "curv": 0.0050, "prior": 0.0010},
        {"name": "curv_high", "range": 0.50, "curv": 0.0200, "prior": 0.0010},
        {"name": "prior_low", "range": 0.50, "curv": 0.0100, "prior": 0.0005},
        {"name": "prior_high", "range": 0.50, "curv": 0.0100, "prior": 0.0020},
        {"name": "gentle_all", "range": 0.25, "curv": 0.0050, "prior": 0.0005},
    ]

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pinode_regularization_sweep.csv"

    # Resume: load any previously completed results
    results = []
    completed_names = set()
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["lambda_kq_range"] = float(row["lambda_kq_range"])
                row["lambda_kq_curvature"] = float(row["lambda_kq_curvature"])
                row["lambda_kq_prior"] = float(row["lambda_kq_prior"])
                row["test_rmse_kw"] = float(row["test_rmse_kw"])
                results.append(row)
                completed_names.add(row["name"])
        print(f"Resumed: found {len(results)} completed configs in {out_csv}")

    print(f"Starting sweep with {len(sweep)} configs ({len(completed_names)} already done)...")
    for i, cfg in enumerate(sweep, 1):
        if cfg["name"] in completed_names:
            print(f"\n[{i}/{len(sweep)}] {cfg['name']} -- SKIPPED (already completed)")
            continue
        print("\n" + "=" * 80)
        print(
            f"[{i}/{len(sweep)}] {cfg['name']} | "
            f"range={cfg['range']}, curv={cfg['curv']}, prior={cfg['prior']}"
        )
        print("=" * 80)

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

        # Per-run regularization override.
        model.LAMBDA_KQ_RANGE = cfg["range"]
        model.LAMBDA_KQ_CURVATURE = cfg["curv"]
        model.LAMBDA_KQ_PRIOR = cfg["prior"]

        train_loader = model.prepare_sequence_dataloader(
            X_tr_seq[:-n_val], X_tr_uns_seq[:-n_val], y_tr_seq[:-n_val], shuffle=True
        )
        val_loader = model.prepare_sequence_dataloader(
            X_tr_seq[-n_val:], X_tr_uns_seq[-n_val:], y_tr_seq[-n_val:], shuffle=False
        )
        test_loader = model.prepare_sequence_dataloader(
            X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False
        )

        ckpt = out_dir / f"best_pinode_reg_{cfg['name']}.pt"
        model.train(
            train_loader,
            val_loader=val_loader,
            live_plot=False,
            checkpoint_path=str(ckpt),
        )
        _, test_rmse = model.evaluate_loader(test_loader)
        print(f"RESULT [{cfg['name']}] TEST RMSE: {test_rmse:.2f} kW")

        results.append(
            {
                "name": cfg["name"],
                "lambda_kq_range": cfg["range"],
                "lambda_kq_curvature": cfg["curv"],
                "lambda_kq_prior": cfg["prior"],
                "test_rmse_kw": round(float(test_rmse), 4),
            }
        )

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    best = min(results, key=lambda r: r["test_rmse_kw"])
    print("\n" + "#" * 80)
    print(f"Best config: {best}")
    print(f"Saved results to: {out_csv}")
    print("#" * 80)


if __name__ == "__main__":
    main()

