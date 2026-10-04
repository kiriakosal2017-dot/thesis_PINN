"""Aggregate the saved predictions into the tables reported in the paper.

    python -m experiments.summarise

Writes ``results_v2/summary.json`` and prints the tables. Neural models are reported as
mean and standard deviation over seeds; the closed-form baselines are deterministic.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

from experiments.train import VARIANTS, load_everything
from pinode import baselines
from pinode.baselines import torque_identity_wake
from pinode.training import metrics

RUNS = Path("results_v2/runs")
TRANSIENT_PERCENTILE = 75


def seed_runs(tag):
    files = sorted(glob.glob(str(RUNS / f"{tag}_seed*.npz")), key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    return [np.load(f) for f in files]


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return [float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0]


def main():
    source, scaler, train, val, test, targets = load_everything()
    truth = {"test": test.y[test.ends], **{f"zs_{k}": t.y[t.ends] for k, (_, _, t) in targets.items()}}
    accel = np.abs(test.accel[test.ends])
    threshold = float(np.percentile(accel, TRANSIENT_PERCENTILE))
    transient = accel >= threshold

    out = {"n_windows": {k: int(len(v)) for k, v in truth.items()},
           "transient_threshold": threshold, "models": {}}

    def row(pred_sets):
        """pred_sets: list (one per seed) of dict-like with the prediction arrays."""
        r = {}
        for key, y in truth.items():
            m = [metrics(p[key], y) for p in pred_sets]
            r[key] = {"rmse": mean_sd([a for a, _ in m]), "mape": mean_sd([b for _, b in m])}
        r["steady_mape"] = mean_sd([metrics(p["test"][~transient], truth["test"][~transient])[1] for p in pred_sets])
        r["transient_mape"] = mean_sd([metrics(p["test"][transient], truth["test"][transient])[1] for p in pred_sets])
        r["n_seeds"] = len(pred_sets)
        return r

    for cls in baselines.ALL:
        f = RUNS / f"baseline_{cls.__name__}.npz"
        if f.exists():
            out["models"][cls.name] = row([np.load(f)])
    for tag in VARIANTS:
        runs = seed_runs(tag)
        if not runs:
            continue
        out["models"][tag] = row(runs)
        info = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RUNS / f"{tag}_seed*.json")))]
        out["models"][tag]["epochs"] = mean_sd([i["epochs"] for i in info])
        if len(runs) > 1:
            ens = {k: np.mean([r[k] for r in runs], axis=0) for k in truth}
            out["models"][tag]["ensemble"] = {k: list(metrics(ens[k], truth[k])) for k in truth}

    # Wake fraction learned by PI-NODE against the torque-identity value implied by the data.
    runs = seed_runs("pinode")
    if runs:
        w_data = torque_identity_wake(test, source)
        w_net = np.mean([r["test_wake"] for r in runs], axis=0)
        draft = test.frame["draft_mid"].values[test.ends]
        laden = draft >= np.nanmedian(train.frame["draft_mid"].values)
        out["wake"] = {
            "net_mean": float(w_net.mean()), "data_mean": float(w_data.mean()),
            "corr": float(np.corrcoef(w_net, w_data)[0, 1]),
            "mae": float(np.mean(np.abs(w_net - w_data))),
            "seed_sd_of_mean": float(np.std([r["test_wake"].mean() for r in runs], ddof=1)),
            "net_laden": float(w_net[laden].mean()), "net_ballast": float(w_net[~laden].mean()),
            "data_laden": float(w_data[laden].mean()), "data_ballast": float(w_data[~laden].mean()),
        }
    out["target_wake_median"] = {k: float(np.median(torque_identity_wake(t, v))) for k, (v, _, t) in targets.items()}
    out["source_wake_median"] = float(np.median(torque_identity_wake(train, source)))

    Path("results_v2/summary.json").write_text(json.dumps(out, indent=1))

    keys = list(truth)
    print(f"{'model':30s} {'seeds':>5s} " + " ".join(f"{k + ' RMSE':>12s} {k + ' MAPE':>12s}" for k in keys) + "   steady/transient MAPE")
    for name, r in out["models"].items():
        cells = " ".join(f"{r[k]['rmse'][0]:7.0f}±{r[k]['rmse'][1]:<4.0f} {r[k]['mape'][0]:6.2f}±{r[k]['mape'][1]:<5.2f}" for k in keys)
        print(f"{name:30s} {r['n_seeds']:5d} {cells}   {r['steady_mape'][0]:.2f} / {r['transient_mape'][0]:.2f}")
    if "wake" in out:
        print("wake:", {k: round(v, 3) for k, v in out["wake"].items()})


if __name__ == "__main__":
    main()
