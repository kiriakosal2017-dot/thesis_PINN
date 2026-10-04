"""Blocked cross-validation in time on the source vessel.

    python -m experiments.blocked_cv --fold 0

The single chronological split puts one operating profile in the test period. To check that
the model ranking does not hinge on that choice, the record is cut into five contiguous
blocks; each block in turn is the test period, and the other four are divided into training
and validation day-blocks exactly as in the main protocol. Windows never straddle a block
boundary.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train import SEQ_LEN, VARIANTS
from pinode import baselines
from pinode.data import (VAL_BLOCK_ROWS, VAL_EVERY, Scaler, Split, _window_ends, load_frame,
                         load_vessel)
from pinode.models import build
from pinode.training import Windows, fit_best_lr, metrics, predict

FOLDS = 5
TAGS = ["pinode", "pinode_gru", "pinode_static", "data", "data_gru", "hybrid", "pikan"]


def fold_splits(frame, fold):
    N = len(frame)
    edges = [int(N * i / FOLDS) for i in range(FOLDS + 1)]
    dt = frame["dt"].values
    # Day-long blocks of the four non-test fifths; every fifth one is validation, as in the
    # main protocol.
    days = []
    for b in range(FOLDS):
        if b != fold:
            cuts = list(range(edges[b], edges[b + 1], VAL_BLOCK_ROWS)) + [edges[b + 1]]
            days += list(zip(cuts[:-1], cuts[1:]))
    val_blocks = [d for i, d in enumerate(days) if i % VAL_EVERY == 2]
    train_blocks = [d for i, d in enumerate(days) if i % VAL_EVERY != 2]
    scaler = Scaler().fit(frame.iloc[np.concatenate([np.arange(*d) for d in train_blocks])])
    X = scaler.transform(frame)

    def split(blocks):
        ends = np.concatenate([_window_ends(dt, lo, hi, SEQ_LEN) for lo, hi in blocks if hi - lo >= SEQ_LEN])
        return Split(X=X, n=frame["n"].values.astype(np.float32), V=frame["V"].values.astype(np.float32),
                     y=frame["power"].values.astype(np.float32), dt=dt.astype(np.float32),
                     accel=frame["accel"].values, frame=frame, ends=ends)

    return scaler, split(train_blocks), split(val_blocks), split([(edges[fold], edges[fold + 1])])


def main(fold):
    torch.set_num_threads(2)
    source = load_vessel("source")
    frame = load_frame(source)
    scaler, train, val, test = fold_splits(frame, fold)
    truth = test.y[test.ends]
    rows = []
    for cls in baselines.ALL:
        rmse, mape = metrics(cls().fit(train, source).predict(test, source), truth)
        rows.append({"fold": fold, "model": cls.name, "rmse": rmse, "mape": mape})
    wtr, wva, wte = (Windows(s, SEQ_LEN) for s in (train, val, test))
    for tag in TAGS:
        kind, kw = VARIANTS[tag]

        def make():
            model = build(kind, source, scaler, **kw)
            if kind in ("hybrid", "pikan"):
                model.calibrate(torch.as_tensor(train.V[train.ends]), torch.as_tensor(train.y[train.ends]))
            return model

        model, info = fit_best_lr(make, wtr, wva, scaler, seed=0)
        rmse, mape = metrics(predict(model, wte), truth)
        rows.append({"fold": fold, "model": tag, "rmse": rmse, "mape": mape, **info})
        print(rows[-1], flush=True)
    Path(f"results_v2/blocked_cv_fold{fold}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    main(ap.parse_args().fold)
