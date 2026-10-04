"""Design search with validation-based selection.

    python -m experiments.search --name pinode-relax-nd-fixed --seed 0

For one configuration and seed: choose the learning rate and stopping epoch on the validation
blocks, then score the model on the source test period and on the transfer vessels.
Configurations are compared by their validation loss only; test numbers are recorded for
reporting, not for choosing.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train import SEQ_LEN, load_everything
from pinode.models import build
from pinode.training import Windows, fit_best_lr, metrics, predict

OUT = Path("results_v2/search")


def configs():
    out = {}
    for enc, inp, kq in itertools.product(("static", "relax", "ode", "gru"), ("raw", "nd", "both"), (False, True)):
        out[f"pinode-{enc}-{inp}-{'trainkq' if kq else 'fixed'}"] = ("pinode", dict(encoder=enc, inputs=inp, trainable_kq=kq))
    for kind, inp, hid in itertools.product(("data", "data_gru"), ("raw", "nd", "both"), ((128, 64, 32), (64, 32))):
        out[f"{kind}-{inp}-{'x'.join(map(str, hid))}"] = (kind, dict(inputs=inp, hidden=hid))
    for kind, inp in itertools.product(("hybrid", "pikan"), ("raw", "both")):
        out[f"{kind}-{inp}"] = (kind, dict(inputs=inp))
    return out


def run(name, seed):
    torch.set_num_threads(2)
    kind, kw = configs()[name]
    source, scaler, train, val, test, targets = load_everything()

    def make():
        model = build(kind, source, scaler, **kw)
        if kind in ("hybrid", "pikan"):
            model.calibrate(torch.as_tensor(train.V[train.ends]), torch.as_tensor(train.y[train.ends]))
        return model

    wtr, wva, wte = (Windows(s, SEQ_LEN) for s in (train, val, test))
    model, info = fit_best_lr(make, wtr, wva, scaler, seed)

    def score(m):
        res = {"test": list(metrics(predict(m, wte), test.y[test.ends]))}
        for label, (vessel, _, t_test) in targets.items():
            if hasattr(m, "propeller"):
                m.propeller.set_propeller(vessel)
            res[f"zs_{label}"] = list(metrics(predict(m, Windows(t_test, SEQ_LEN)), t_test.y[t_test.ends]))
        if hasattr(m, "propeller"):
            m.propeller.set_propeller(source)
        return res

    result = {"name": name, "seed": seed, **info, "val_rmse": metrics(predict(model, wva), val.y[val.ends])[0],
              "scores": score(model)}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / f"{name}_seed{seed}.pt")
    (OUT / f"{name}_seed{seed}.json").write_text(json.dumps(result, indent=1))
    print(name, seed, info, result["scores"]["test"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(configs()))
    else:
        run(a.name, a.seed)
