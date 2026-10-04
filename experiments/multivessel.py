"""Leave-one-vessel-out: train on four ships, test on the fifth without any of its power data.

    python -m experiments.multivessel --holdout C --tag pinode --seed 0
    python -m experiments.multivessel --holdout C --baselines

Every vessel contributes the first 80 % of its record to training (with every fifth day-long
block held out for validation, as in the main protocol). The held-out vessel is scored on the
last 20 % of its record, the same rows used for single-source zero-shot transfer, so the two
settings can be compared directly. Each row carries the particulars of its own propeller, so
one PI-NODE serves all ships; the black-box models see the same pooled data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.train import MAIN, VARIANTS
from pinode.baselines import torque_identity_wake
from pinode.bseries import open_water_cubics
from pinode.data import (VAL_BLOCK_ROWS, VAL_EVERY, Scaler, Split, _window_ends, load_frame,
                         load_vessel)
from pinode.models import PropellerLayer, build
from pinode.training import Windows, fit_best_lr, metrics, predict

SEQ_LEN = 10
OUT = Path("results_v2/multivessel")
EXTRA = {"data_both": ("data", dict(inputs="both")), "data_gru_both": ("data_gru", dict(inputs="both"))}
TAGS = ["pinode", "pinode_gru", "pinode_static", "pinode_relax", "data", "data_both", "data_gru", "data_gru_both", "hybrid"]


def fleet():
    from ship_registry import SOURCE_VESSEL, TRANSFER_VESSELS
    return {"S": SOURCE_VESSEL, **TRANSFER_VESSELS}


def load(holdout):
    names = fleet()
    vessels = {k: load_vessel(v) for k, v in names.items()}
    frames = {k: load_frame(v) for k, v in vessels.items()}
    offsets, start = {}, 0
    for k, f in frames.items():
        offsets[k] = start
        start += len(f)
    frame = pd.concat(frames.values(), ignore_index=True)
    dt = frame["dt"].values.copy()
    for k in frames:
        dt[offsets[k]] = 1e9          # a window never spans two vessels

    train_blocks, val_blocks, test_block = [], [], None
    for k, f in frames.items():
        lo, cut, hi = offsets[k], offsets[k] + int(len(f) * 0.8), offsets[k] + len(f)
        if k == holdout:
            test_block = (cut, hi)
            continue
        edges = list(range(lo, cut, VAL_BLOCK_ROWS)) + [cut]
        for i, b in enumerate(zip(edges[:-1], edges[1:])):
            (val_blocks if i % VAL_EVERY == 2 else train_blocks).append(b)
    scaler = Scaler().fit(frame.iloc[np.concatenate([np.arange(*b) for b in train_blocks])])
    X = scaler.transform(frame)

    def split(blocks):
        ends = np.concatenate([_window_ends(dt, lo, hi, SEQ_LEN) for lo, hi in blocks if hi - lo >= SEQ_LEN])
        return Split(X=X, n=frame["n"].values.astype(np.float32), V=frame["V"].values.astype(np.float32),
                     y=frame["power"].values.astype(np.float32), dt=frame["dt"].values.astype(np.float32),
                     accel=frame["accel"].values, frame=frame, ends=ends)

    # Per-row propeller particulars.
    N = len(frame)
    D, kq, jmax = np.zeros(N, np.float32), np.zeros((N, 4), np.float32), np.zeros(N, np.float32)
    for k, f in frames.items():
        rows = slice(offsets[k], offsets[k] + len(f))
        v = vessels[k]
        D[rows] = v.D
        kq[rows] = open_water_cubics(v.p_d, v.ae_a0, v.z)[1]
        jmax[rows] = float(PropellerLayer(v).j_max)
    prop = {"D": torch.as_tensor(D), "kq": torch.as_tensor(kq), "j_max": torch.as_tensor(jmax)}
    return vessels, frames, offsets, scaler, split(train_blocks), split(val_blocks), split([test_block]), prop


def windows(split, prop):
    w = Windows(split, SEQ_LEN)
    w.prop = prop
    return w


def run(holdout, tag, seed):
    torch.set_num_threads(2)
    vessels, frames, offsets, scaler, train, val, test, prop = load(holdout)
    kind, kw = {**VARIANTS, **EXTRA}[tag]
    anchor = vessels[holdout]

    def make():
        model = build(kind, anchor, scaler, **kw)
        if kind in ("hybrid", "pikan"):
            model.calibrate(torch.as_tensor(train.V[train.ends]), torch.as_tensor(train.y[train.ends]))
        return model

    model, info = fit_best_lr(make, windows(train, prop), windows(val, prop), scaler, seed)
    rmse, mape = metrics(predict(model, windows(test, prop)), test.y[test.ends])
    vr = metrics(predict(model, windows(val, prop)), val.y[val.ends])[0]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"holdout": holdout, "tag": tag, "seed": seed, **info, "val_rmse": vr, "rmse": rmse, "mape": mape}
    (OUT / f"{holdout}_{tag}_seed{seed}.json").write_text(json.dumps(res))
    print(res, flush=True)


def run_baselines(holdout):
    """Closed-form transfer: pooled constant K_Q (propeller law) and pooled constant wake."""
    vessels, frames, offsets, scaler, train, val, test, prop = load(holdout)
    rho = 1025.0
    base = lambda n, D: 2 * np.pi * rho * n ** 3 * D ** 5 / 1000.0
    rows = train.ends
    b = base(train.n[rows].astype(float), prop["D"].numpy()[rows].astype(float))
    kq_const = float(b @ train.y[rows] / (b @ b))
    wakes = []
    for k, f in frames.items():
        if k == holdout:
            continue
        lo, hi = offsets[k], offsets[k] + int(len(f) * 0.8)
        sub = Split(X=train.X, n=train.n, V=train.V, y=train.y, dt=train.dt, accel=train.accel, frame=train.frame,
                    ends=train.ends[(train.ends >= lo) & (train.ends < hi)])
        wakes.append(float(np.median(torque_identity_wake(sub, vessels[k]))))
    w = float(np.mean(wakes))
    v = vessels[holdout]
    n, V, y = test.n[test.ends].astype(float), test.V[test.ends].astype(float), test.y[test.ends]
    layer = PropellerLayer(v)
    p_wake = layer(torch.full((len(n),), w), torch.as_tensor(n, dtype=torch.float32), torch.as_tensor(V, dtype=torch.float32)).numpy()
    out = []
    for tag, pred in (("Propeller law (pooled K_Q)", kq_const * base(n, v.D)), ("B-series, pooled constant wake", p_wake)):
        rmse, mape = metrics(pred, y)
        out.append({"holdout": holdout, "tag": tag, "seed": 0, "rmse": rmse, "mape": mape})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{holdout}_baselines.json").write_text(json.dumps(out))
    print(out, "train wakes", wakes, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    run_baselines(a.holdout) if a.baselines else run(a.holdout, a.tag, a.seed)
