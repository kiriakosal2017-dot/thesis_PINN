"""One training and evaluation protocol for every neural model in the comparison."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn

MAX_EPOCHS = 400
PATIENCE = 40
BATCH = 256
LR = 1e-4
# Learning rates tried for every model; the one with the lowest validation loss is kept.
LR_GRID = (1e-4, 3e-5)
# One-minute records make consecutive windows near-duplicates (they share all but one row), so
# an epoch draws one window in EPOCH_STRIDE at random: each record is then seen about once.
EPOCH_STRIDE = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Windows:
    """A split held on the device, gathered into windows by end index.

    By default a window is the ``seq_len`` consecutive rows ending at the target row. With
    ``span`` set, the window is instead ``seq_len`` rows drawn at random from the last
    ``span`` rows (the target row always included), which imitates sparse, irregular
    logging; ``dt`` then carries the real time elapsed between the rows that were kept.
    """

    def __init__(self, split, seq_len, device="cpu", span=None, seed=0, step=1):
        self.X = torch.as_tensor(split.X, device=device)
        self.dt = torch.as_tensor(split.dt, device=device)
        self.time = torch.cumsum(self.dt.double(), 0)
        self.n = torch.as_tensor(split.n, device=device)
        self.V = torch.as_tensor(split.V, device=device)
        self.y = torch.as_tensor(split.y, device=device)
        self.ends = torch.as_tensor(split.ends, device=device)
        self.seq_len, self.span, self.seed = seq_len, span, seed
        # step > 1 spaces the rows of a regular window ``step`` records apart (longer history).
        self.step = step
        self.offsets = torch.arange(-(seq_len - 1) * step, 1, step, device=device)
        self.reseed()

    def reseed(self):
        self.gen = torch.Generator().manual_seed(self.seed)

    def batch(self, ends):
        if self.span is None:
            rows = ends[:, None] + self.offsets[None, :]
            if self.step == 1:
                dt = self.dt[rows]
            else:
                t = self.time[rows]
                dt = torch.cat([t.new_zeros(len(ends), 1), t[:, 1:] - t[:, :-1]], dim=1).float()
        else:
            # seq_len - 1 distinct earlier rows per sample, in chronological order.
            scores = torch.rand(len(ends), self.span - 1, generator=self.gen)
            back = scores.topk(self.seq_len - 1, dim=1).indices + 1
            back = torch.sort(back, dim=1, descending=True).values
            rows = torch.cat([ends[:, None] - back, ends[:, None]], dim=1)
            t = self.time[rows]
            dt = torch.cat([t.new_zeros(len(ends), 1), t[:, 1:] - t[:, :-1]], dim=1).float()
        out = {"x": self.X[rows], "dt": dt, "n": self.n[ends], "V": self.V[ends]}
        # Per-row propeller particulars, set by the multi-vessel experiments.
        for key, values in getattr(self, "prop", {}).items():
            out[key] = values[ends]
        return out, self.y[ends]


@torch.no_grad()
def predict(model, windows, ends=None, chunk=8192):
    """Predicted power (kW) for every window, in order."""
    model.eval()
    windows.reseed()
    ends = windows.ends if ends is None else ends
    out = [model(windows.batch(ends[i:i + chunk])[0]) for i in range(0, len(ends), chunk)]
    return torch.cat(out).cpu().numpy()


def metrics(pred, true):
    """RMSE (kW) and MAPE (%); the MAPE denominator is floored at 100 kW."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mape = float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 100.0)) * 100.0)
    return rmse, mape


def fit(model, train, val, scaler, seed, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        batch_size=BATCH, verbose=False):
    """Adam + Huber loss on standardised power; the best validation epoch is restored.

    Early stopping and the learning-rate schedule both watch the validation Huber loss, so
    the test period plays no part in any training decision.
    """
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=lr * 0.05)
    huber = nn.SmoothL1Loss(beta=0.5)
    extra = getattr(model, "regulariser", None)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = train.ends[torch.randperm(len(train.ends), generator=gen).to(train.ends.device)]
        perm = perm[: max(len(perm) // EPOCH_STRIDE, batch_size)]
        for i in range(0, len(perm), batch_size):
            batch, y = train.batch(perm[i:i + batch_size])
            power = model(batch)
            loss = huber((power - scaler.y_mean) / scaler.y_std, (y - scaler.y_mean) / scaler.y_std)
            if extra is not None:
                loss = loss + extra(batch, power)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        pv = torch.as_tensor(predict(model, val))
        val_loss = float(huber((pv - scaler.y_mean) / scaler.y_std, (val.y[val.ends].cpu() - scaler.y_mean) / scaler.y_std))
        sched.step(val_loss)
        if val_loss < best - 1e-5:
            best, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if verbose:
            print(f"  epoch {epoch + 1:3d}  val {val_loss:.5f}  best {best:.5f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return {"epochs": best_epoch, "val_loss": best}


def fit_best_lr(make_model, train, val, scaler, seed, lr_grid=LR_GRID):
    """Train one model per learning rate and keep the one with the lowest validation loss.

    The operating profile drifts between the training and validation periods, so the best
    step size differs between model families; choosing it on the validation period gives
    every family the same chance.
    """
    best = None
    for lr in lr_grid:
        set_seed(seed)
        model = make_model()
        info = fit(model, train, val, scaler, seed, lr=lr)
        if best is None or info["val_loss"] < best[1]["val_loss"]:
            best = (model, {**info, "lr": lr})
    return best


def refit(model, trainval, scaler, seed, lr, epochs, batch_size=BATCH):
    """Train for a fixed number of epochs on training + validation rows.

    Used after ``fit_best_lr`` has chosen the learning rate and the stopping epoch on the
    validation period: the final model then learns from that period too. The learning-rate
    halvings that the schedule made before the best epoch are not replayed.
    """
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    huber = nn.SmoothL1Loss(beta=0.5)
    extra = getattr(model, "regulariser", None)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = trainval.ends[torch.randperm(len(trainval.ends), generator=gen)]
        perm = perm[: max(len(perm) // EPOCH_STRIDE, batch_size)]
        for i in range(0, len(perm), batch_size):
            batch, y = trainval.batch(perm[i:i + batch_size])
            power = model(batch)
            loss = huber((power - scaler.y_mean) / scaler.y_std, (y - scaler.y_mean) / scaler.y_std)
            if extra is not None:
                loss = loss + extra(batch, power)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    return model
