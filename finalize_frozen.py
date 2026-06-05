"""
Finalize the `frozen_bseries` ablation config from its saved best checkpoint
WITHOUT further training. The config had plateaued (~200 epochs), so we evaluate
the best checkpoint on the test set — exactly as run_config() would after training
(which restores best_state anyway) — and append the row to ablation_results.csv.
"""
import csv
import sys
import torch
from pathlib import Path

from main_PI_NODE_Propeller import PINODEPropellerModel
from pinode_common import load_danae_temporal_sequences, make_loaders, predict_power, rmse_mape
from ablation_study import CONFIGS, LAMBDA_RANGE, LAMBDA_CURV, LAMBDA_PRIOR

name = sys.argv[1] if len(sys.argv) > 1 else "frozen_bseries"
cfg = CONFIGS[name]
results_dir = Path("results")
ckpt = results_dir / f"ablation_{name}.pt"

print(f"Loading data for {name} (meta_exclude={cfg['meta_exclude']}) ...")
proc, feature_indices, calm_idx, weather_idx, train_tuple, test_tuple = \
    load_danae_temporal_sequences(meta_exclude=cfg["meta_exclude"])

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
model.LAMBDA_KQ_RANGE = LAMBDA_RANGE
model.LAMBDA_KQ_CURVATURE = LAMBDA_CURV
model.LAMBDA_KQ_PRIOR = LAMBDA_PRIOR

print(f"Loading best checkpoint: {ckpt}")
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
