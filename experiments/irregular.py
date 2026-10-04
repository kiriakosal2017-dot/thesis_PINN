"""Sparse, irregular logging: does integrating over real elapsed time pay off?

    python -m experiments.irregular --tag pinode_nd --seed 0

Each window keeps ten rows. Under regular logging they are the last ten minutes; under sparse
logging they are ten rows drawn at random from the last 30 or 60 minutes, so the intervals
between them vary. Two versions of every model are scored on all three conditions: the one
trained on regular windows (the main checkpoint), and one trained with random 10-of-30
windows. An encoder that ignores elapsed time sees the same ten rows either way.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train import OUT, SEQ_LEN, load_everything, make_model
from pinode.data import MAX_GAP_S, Split, _window_ends
from pinode.training import Windows, fit_best_lr, metrics, predict

MAX_SPAN = 60
CONDITIONS = {"regular": None, "sparse30": 30, "sparse60": 60}
DESIGN_VESSELS = ("A", "B")


def long_ends(split):
    """Window ends that have MAX_SPAN gap-free rows behind them, so all conditions share targets."""
    lo, hi = int(split.ends.min()) - SEQ_LEN + 1, int(split.ends.max()) + 1
    ends = _window_ends(split.dt, lo, hi, MAX_SPAN)
    return Split(X=split.X, n=split.n, V=split.V, y=split.y, dt=split.dt, accel=split.accel,
                 frame=split.frame, ends=ends)


def build(tag, source, scaler, train):
    model = make_model(tag, source, scaler, train)
    if hasattr(model, "encoder"):
        model.encoder.substeps = 4      # intervals reach tens of minutes here
    return model


def score(model, splits, vessels, source):
    out = {}
    for name, split in splits.items():
        if hasattr(model, "propeller"):
            model.propeller.set_propeller(vessels.get(name, source))
        for cond, span in CONDITIONS.items():
            pred = predict(model, Windows(split, SEQ_LEN, span=span, seed=123))
            out[f"{name}/{cond}"] = list(metrics(pred, split.y[split.ends]))
    if hasattr(model, "propeller"):
        model.propeller.set_propeller(source)
    return out


def main(tag, seed):
    torch.set_num_threads(2)
    source, scaler, train, val, test, targets = load_everything()
    train, val, test = long_ends(train), long_ends(val), long_ends(test)
    splits = {"test": test, **{k: long_ends(targets[k][2]) for k in DESIGN_VESSELS if k in targets}}
    vessels = {k: targets[k][0] for k in DESIGN_VESSELS if k in targets}

    regular = build(tag, source, scaler, train)
    regular.load_state_dict(torch.load(OUT / f"{tag}_seed{seed}.pt"), strict=False)
    result = {"tag": tag, "seed": seed, "trained_regular": score(regular, splits, vessels, source)}

    model, info = fit_best_lr(lambda: build(tag, source, scaler, train),
                              Windows(train, SEQ_LEN, span=30, seed=seed),
                              Windows(val, SEQ_LEN, span=30, seed=seed + 1000), scaler, seed)
    result["trained_sparse"] = score(model, splits, vessels, source)
    result["info"] = info
    Path("results_v2/irregular").mkdir(exist_ok=True)
    Path(f"results_v2/irregular/{tag}_seed{seed}.json").write_text(json.dumps(result, indent=1))
    print(tag, seed, info, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.tag, a.seed)
