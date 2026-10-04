"""Few-shot adaptation: fine-tune source-trained models on the first part of a target vessel.

    python -m experiments.fewshot --seed 0

For each transfer vessel and each data budget (a fraction of the vessel's first 80 %),
the source checkpoint is fine-tuned and then scored on the vessel's last 20 %. The last
fifth of the budget is held out for early stopping. The physics-only baseline is adapted
the same way, by re-estimating its single wake constant from the budget.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train import OUT, SEQ_LEN, load_everything, make_model
from pinode.baselines import BSeriesConstantWake
from pinode.data import Split
from pinode.training import Windows, fit, metrics, predict

FRACTIONS = [0.01, 0.05, 0.10, 0.25]
# (checkpoint tag, adaptation mode). 'full' fine-tunes every weight; 'offset' keeps the
# network frozen and re-estimates only the scalar wake offset of the new vessel.
RUNS = [("pinode", "full"), ("pinode_static", "full"), ("data", "full"), ("data_gru", "full"), ("hybrid", "full")]


def subset(split, ends):
    return Split(X=split.X, n=split.n, V=split.V, y=split.y, dt=split.dt, accel=split.accel,
                 frame=split.frame, ends=ends)


def adapt(model, mode, ft_train, ft_val, scaler, seed):
    if mode == "offset":
        for name, p in model.named_parameters():
            p.requires_grad_(name == "wake_offset")
        fit(model, Windows(ft_train, SEQ_LEN), Windows(ft_val, SEQ_LEN), scaler, seed,
            lr=2e-2, max_epochs=300, patience=30, batch_size=64)
    else:
        fit(model, Windows(ft_train, SEQ_LEN), Windows(ft_val, SEQ_LEN), scaler, seed,
            lr=1e-4, max_epochs=300, patience=30, batch_size=64)


def main(seed, runs, vessels, out_name):
    torch.set_num_threads(2)
    source, scaler, train, val, test, targets = load_everything()
    rows = []
    for label, (vessel, pool, t_test) in targets.items():
        if vessels and label not in vessels:
            continue
        truth = t_test.y[t_test.ends]
        wtest = Windows(t_test, SEQ_LEN)
        for frac in FRACTIONS:
            budget = pool.ends[: max(int(len(pool.ends) * frac), 20)]
            n_val = max(len(budget) // 5, 5)
            ft_train, ft_val = subset(pool, budget[:-n_val]), subset(pool, budget[-n_val:])

            phys = BSeriesConstantWake().fit(subset(pool, budget), vessel)
            rmse, mape = metrics(phys.predict(t_test, vessel), truth)
            rows.append({"vessel": label, "frac": frac, "n": int(len(budget)), "model": "bseries_const_wake",
                         "seed": seed, "rmse": rmse, "mape": mape})

            for tag, mode in runs:
                model = make_model(tag, source, scaler, train)
                model.load_state_dict(torch.load(OUT / f"{tag}_seed{seed}.pt"), strict=False)
                if hasattr(model, "propeller"):
                    model.propeller.set_propeller(vessel)
                adapt(model, mode, ft_train, ft_val, scaler, seed)
                rmse, mape = metrics(predict(model, wtest), truth)
                rows.append({"vessel": label, "frac": frac, "n": int(len(budget)), "model": f"{tag}:{mode}",
                             "seed": seed, "rmse": rmse, "mape": mape})
                print(rows[-1], flush=True)
    Path(f"results_v2/{out_name}_seed{seed}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="*", help="tag:mode pairs, e.g. pinode_relax_nd:offset")
    ap.add_argument("--vessels", nargs="*", help="restrict to these vessel labels")
    ap.add_argument("--out", default="fewshot")
    a = ap.parse_args()
    main(a.seed, [tuple(r.split(":")) for r in a.runs] if a.runs else RUNS, a.vessels, a.out)
