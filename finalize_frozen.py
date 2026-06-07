"""
Evaluate a saved ablation checkpoint on the held-out test set without further training
and upsert the result row into ablation_results.csv.

This is needed when a training run was interrupted after the best checkpoint was saved
but before the evaluation/CSV-write step completed.  Reconstruction is safe because the
training loop always restores best_state before evaluating, so loading the checkpoint
directly reproduces the same metric.
"""
import csv
import sys
import torch
from pathlib import Path

from main_PI_NODE_Propeller import PINODEPropellerModel
from pinode_common import load_danae_temporal_sequences, make_loaders, predict_power, rmse_mape
from ablation_study import CONFIGS, LAMBDA_RANGE, LAMBDA_CURV, LAMBDA_PRIOR

# Allow the target config name to be overridden via CLI; default is frozen_bseries.
name = sys.argv[1] if len(sys.argv) > 1 else "frozen_bseries"
cfg = CONFIGS[name]
results_dir = Path("results")
ckpt = results_dir / f"ablation_{name}.pt"

print(f"Loading data for {name} (meta_exclude={cfg['meta_exclude']}) ...")
proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple = \
    load_danae_temporal_sequences(meta_exclude=cfg["meta_exclude"])

# epochs=1 prevents any accidental gradient updates; the model is used purely for
# inference after the checkpoint is loaded.
model = PINODEPropellerModel(
    input_size=train_tuple[0].shape[2],
    feature_indices=feature_indices,
    calm_water_indices=calm_idx,
    weather_indices=weather_idx,
    data_processor=proc,
    hidden_size=64,
    ode_num_layers=2,
    lr=0.001,
    epochs=1,
    batch_size=64,
    loss_function_choice="SmoothL1",
    encoder_mode=cfg["encoder_mode"],
    use_ode=cfg["use_ode"],
    use_weather=cfg["use_weather"],
    freeze_polynomials=cfg["freeze_polynomials"],
)
# Regularisation lambdas must match the training run so the physics loss scale is
# consistent with what was used to select the best checkpoint.
model.LAMBDA_KQ_RANGE = LAMBDA_RANGE
model.LAMBDA_KQ_CURVATURE = LAMBDA_CURV
model.LAMBDA_KQ_PRIOR = LAMBDA_PRIOR

print(f"Loading best checkpoint: {ckpt}")
# map_location="cpu" avoids device mismatches when the checkpoint was saved on GPU.
state = torch.load(ckpt, map_location="cpu")
model.model.load_state_dict(state)

_, _, test_loader = make_loaders(model, train_tuple, test_tuple)
preds, true = predict_power(model, test_loader)
rmse, mape = rmse_mape(preds, true)
row = {
    "config": name,
    "calm_features": len(calm_idx),
    "weather_features": len(weather_idx),
    "test_rmse_kw": round(rmse, 2),
    "test_mape_pct": round(mape, 3),
}
print(f"\n[{name}] Test RMSE = {rmse:.2f} kW | MAPE = {mape:.2f}%  -> {row}")

# Upsert: read existing rows, drop any stale entry for this config, append the fresh one.
out_csv = results_dir / "ablation_results.csv"
rows = []
if out_csv.exists():
    with open(out_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["config"] != name]
rows.append(row)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {out_csv} ({len(rows)} configs)")
