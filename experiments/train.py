"""Train one model on the source vessel and evaluate it everywhere.

    python -m experiments.train --tag pinode --seed 0
    python -m experiments.train --tag pinode_gru --seed 0
    python -m experiments.train --baselines

Each run writes ``results_v2/runs/<tag>_seed<k>.pt`` (weights) and ``.npz`` (predictions on
the source validation and test periods and on the test period of every transfer vessel).
Validation predictions are kept because conformal calibration uses them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pinode import baselines
from pinode.data import load_source, load_target
from pinode.models import build
from pinode.training import Windows, fit_best_lr, predict

SEQ_LEN = 10
OUT = Path("results_v2/runs")

# tag -> (model kind, constructor arguments). The configuration of each family was chosen by
# validation error in experiments/search.py; "pinode" is the proposed model and the other
# pinode_* entries change one thing at a time.
MAIN = dict(encoder="ode", inputs="both", trainable_kq=True)
VARIANTS = {
    "pinode":           ("pinode", MAIN),
    "pinode_gru":       ("pinode", {**MAIN, "encoder": "gru"}),
    "pinode_static":    ("pinode", {**MAIN, "encoder": "static"}),
    "pinode_relax":     ("pinode", {**MAIN, "encoder": "relax"}),
    "pinode_fixedkq":   ("pinode", {**MAIN, "trainable_kq": False}),
    "pinode_raw":       ("pinode", {**MAIN, "inputs": "raw"}),
    "pinode_nd":        ("pinode", {**MAIN, "inputs": "nd"}),
    "pinode_noweather": ("pinode", {**MAIN, "use_weather": False}),
    "data":             ("data", {}),
    "data_gru":         ("data_gru", {}),
    "hybrid":           ("hybrid", {}),
    "pikan":            ("pikan", dict(inputs="both")),
}


def transfer_vessels():
    try:
        from ship_registry import TRANSFER_VESSELS
    except ImportError:
        return {}
    return TRANSFER_VESSELS


def load_everything():
    source, scaler, train, val, test = load_source(seq_len=SEQ_LEN)
    targets = {label: load_target(name, scaler, SEQ_LEN) for label, name in transfer_vessels().items()}
    return source, scaler, train, val, test, targets


def make_model(tag, source, scaler, train):
    kind, kw = VARIANTS[tag]
    model = build(kind, source, scaler, **kw)
    if kind in ("hybrid", "pikan"):
        rows = train.ends
        model.calibrate(torch.as_tensor(train.V[rows]), torch.as_tensor(train.y[rows]))
    return model


def run_neural(tag, seed):
    torch.set_num_threads(2)
    source, scaler, train, val, test, targets = load_everything()
    wtr, wva, wte = (Windows(s, SEQ_LEN) for s in (train, val, test))
    model, info = fit_best_lr(lambda: make_model(tag, source, scaler, train), wtr, wva, scaler, seed)

    preds = {"val": predict(model, wva), "test": predict(model, wte)}
    if hasattr(model, "wake"):
        with torch.no_grad():
            model.eval()
            preds["test_wake"] = torch.cat([model.wake(wte.batch(wte.ends[i:i + 8192])[0])
                                            for i in range(0, len(wte.ends), 8192)]).numpy()
    for label, (vessel, _, t_test) in targets.items():
        if hasattr(model, "propeller"):
            model.propeller.set_propeller(vessel)
        preds[f"zs_{label}"] = predict(model, Windows(t_test, SEQ_LEN))
    if hasattr(model, "propeller"):
        model.propeller.set_propeller(source)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{tag}_seed{seed}"
    torch.save(model.state_dict(), f"{stem}.pt")
    np.savez_compressed(f"{stem}.npz", **preds)
    extra = {"hull_k": model.k} if hasattr(model, "k") else {}
    Path(f"{stem}.json").write_text(json.dumps({"tag": tag, "seed": seed, **info, **extra}))
    print(f"{tag} seed {seed}: {info}", flush=True)


def run_baselines():
    source, scaler, train, val, test, targets = load_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    for cls in baselines.ALL:
        model = cls().fit(train, source)
        preds = {"val": model.predict(val, source), "test": model.predict(test, source)}
        for label, (vessel, _, t_test) in targets.items():
            preds[f"zs_{label}"] = model.predict(t_test, vessel)
        np.savez_compressed(OUT / f"baseline_{cls.__name__}.npz", **preds)
        print("fitted", cls.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()
    if args.baselines:
        run_baselines()
    else:
        run_neural(args.tag, args.seed)
