"""Further search on the temporal encoder of PI-NODE: training length, history length, size.

    python -m experiments.ode_search --mode long   --encoder ode --seed 0
    python -m experiments.ode_search --mode design --encoder ode --step 3 --hidden 64 --seed 0

``long``   trains for a fixed, large number of epochs with no early stopping and records the
           validation and test error along the way, to check whether an encoder that looks
           finished after ~100 epochs improves later.
``design`` runs the normal protocol for one (encoder, history spacing, hidden size). Ten rows are
           always used; ``step`` spaces them 1, 3 or 6 minutes apart (10, 30 or 60 minutes of
           history). All settings share the same targets: windows that have 55 gap-free rows
           behind them inside their block.

Everything is compared on validation error; the transfer vessels C and D are not scored here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.train import MAIN
from pinode.data import load_source, load_target
from pinode.models import build
from pinode.training import BATCH, EPOCH_STRIDE, Windows, fit_best_lr, metrics, predict, set_seed

ROWS = 10
LONG_SPAN = 55            # (ROWS - 1) * 6 + 1
DESIGN_VESSELS = {"A": None, "B": None}
OUT = Path("results_v2/ode_search")


def load(step_for_ends=True):
    source, scaler, train, val, test = load_source(seq_len=LONG_SPAN)
    from ship_registry import TRANSFER_VESSELS
    targets = {k: load_target(TRANSFER_VESSELS[k], scaler, LONG_SPAN) for k in DESIGN_VESSELS}
    return source, scaler, train, val, test, targets


def make(source, scaler, encoder, hidden, step):
    model = build("pinode", source, scaler, **{**MAIN, "encoder": encoder, "hidden": hidden})
    model.encoder.substeps = max(1, step)
    return model


def score(model, source, test, targets, step):
    out = {"test": list(metrics(predict(model, Windows(test, ROWS, step=step)), test.y[test.ends]))}
    for k, (vessel, _, t_test) in targets.items():
        model.propeller.set_propeller(vessel)
        out[f"zs_{k}"] = list(metrics(predict(model, Windows(t_test, ROWS, step=step)), t_test.y[t_test.ends]))
    model.propeller.set_propeller(source)
    return out


def design(encoder, hidden, step, seed):
    source, scaler, train, val, test, targets = load()
    wtr, wva = Windows(train, ROWS, step=step), Windows(val, ROWS, step=step)
    model, info = fit_best_lr(lambda: make(source, scaler, encoder, hidden, step), wtr, wva, scaler, seed)
    res = {"encoder": encoder, "hidden": hidden, "step": step, "seed": seed, **info,
           "val_rmse": metrics(predict(model, wva), val.y[val.ends])[0], **score(model, source, test, targets, step)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"design_{encoder}_h{hidden}_s{step}_seed{seed}.json").write_text(json.dumps(res))
    print(res, flush=True)


def long(encoder, seed, epochs, lr):
    source, scaler, train, val, test, targets = load()
    wtr, wva, wte = (Windows(s, ROWS) for s in (train, val, test))
    set_seed(seed)
    model = make(source, scaler, encoder, 64, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
    huber = nn.SmoothL1Loss(beta=0.5)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = wtr.ends[torch.randperm(len(wtr.ends), generator=gen)][: len(wtr.ends) // EPOCH_STRIDE]
        for i in range(0, len(perm), BATCH):
            b, y = wtr.batch(perm[i:i + BATCH])
            p = model(b)
            loss = huber((p - scaler.y_mean) / scaler.y_std, (y - scaler.y_mean) / scaler.y_std) + model.regulariser(b, p)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if ep % 10 == 0 or ep <= 10:
            tr_ends = wtr.ends[:: 20]
            curve.append({"epoch": ep,
                          "train": metrics(predict(model, wtr, tr_ends), train.y[train.ends[::20]])[0],
                          "val": metrics(predict(model, wva), val.y[val.ends])[0],
                          "test": metrics(predict(model, wte), test.y[test.ends])[0]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"long_{encoder}_lr{lr:g}_seed{seed}.json").write_text(json.dumps({"encoder": encoder, "seed": seed, "lr": lr, "curve": curve}))
    best = min(curve, key=lambda c: c["val"])
    print(encoder, seed, lr, "best val", best, "last", curve[-1], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["long", "design"], required=True)
    ap.add_argument("--encoder", default="ode")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.mode == "long":
        long(a.encoder, a.seed, a.epochs, a.lr)
    else:
        design(a.encoder, a.hidden, a.step, a.seed)
