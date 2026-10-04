"""Shaft-power models sharing one interface: ``forward(batch) -> power in kW``.

``batch`` is a dict with
    x  : (B, L, F) scaled feature window, oldest row first
    dt : (B, L)    seconds between consecutive rows (dt[:, 0] is unused)
    n  : (B,)      shaft speed at the last row, rev/s
    V  : (B,)      speed through water at the last row, m/s

PI-NODE predicts one physical quantity, the effective wake fraction w, and turns it into power
with the vessel's own Wageningen B-series torque characteristic. Everything downstream of w is
closed-form, so swapping the propeller particulars is all it takes to move to another ship.
"""
import math

import torch
import torch.nn as nn

from pinode.kan_layer import KAN
from pinode.bseries import open_water_cubics
from pinode.data import (ALL_FEATURES, CALM_FEATURES, FEATURES, ND_CALM_FEATURES, ND_WEATHER_FEATURES,
                         NOMINAL_DT_S, RHO, WEATHER_FEATURES)

# Column positions of each input set within the scaled feature matrix.
INPUT_SETS = {
    "raw": ([ALL_FEATURES.index(f) for f in CALM_FEATURES], [ALL_FEATURES.index(f) for f in WEATHER_FEATURES]),
    "nd": ([ALL_FEATURES.index(f) for f in ND_CALM_FEATURES], [ALL_FEATURES.index(f) for f in ND_WEATHER_FEATURES]),
}
INPUT_SETS["both"] = (INPUT_SETS["raw"][0] + INPUT_SETS["nd"][0], INPUT_SETS["raw"][1] + INPUT_SETS["nd"][1])
N_RAW = len(FEATURES)

W_MIN, W_MAX = 0.0, 0.75


def mlp(sizes, act=nn.ReLU, dropout=0.0):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class PropellerLayer(nn.Module):
    """Torque identity: P = 2*pi*rho*n^3*D^5 * K_Q(J), J = V(1-w)/(nD).

    K_Q is the open-water cubic of the vessel's propeller. It is fixed by default. With
    ``trainable=True`` a learnable offset is added to the four coefficients (the earlier
    PI-NODE formulation, kept as an ablation); the offset is carried along when the
    propeller is swapped.
    """

    def __init__(self, vessel, trainable=False):
        super().__init__()
        self.register_buffer("kq", torch.zeros(4))
        self.register_buffer("D", torch.tensor(1.0))
        self.register_buffer("j_max", torch.tensor(1.0))
        self.delta = nn.Parameter(torch.zeros(4), requires_grad=trainable)
        self.set_propeller(vessel)

    def set_propeller(self, vessel):
        _, kq = open_water_cubics(vessel.p_d, vessel.ae_a0, vessel.z)
        self.kq.copy_(torch.tensor(kq, dtype=torch.float32))
        self.D.fill_(vessel.D)
        # Largest J at which the open-water torque is still positive; beyond it the
        # propeller would be windmilling, which the filtered operating data never contain.
        grid = torch.linspace(0.0, 1.6, 1601)
        vals = self.kq[0].cpu() + grid * (self.kq[1].cpu() + grid * (self.kq[2].cpu() + grid * self.kq[3].cpu()))
        neg = torch.nonzero(vals <= 1e-4)
        self.j_max.fill_(float(grid[neg[0, 0] - 1]) if len(neg) else 1.6)

    def k_q(self, J):
        c = self.kq + self.delta
        return c[0] + J * (c[1] + J * (c[2] + J * c[3]))

    def forward(self, w, n, V, prop=None):
        """``prop`` optionally gives per-sample propellers (D, kq, j_max) for mixed-vessel batches."""
        if prop is None:
            J = torch.minimum(V * (1.0 - w) / (n * self.D), self.j_max)
            return 2.0 * math.pi * RHO * n ** 3 * self.D ** 5 * self.k_q(J) / 1000.0
        D = prop["D"]
        J = torch.minimum(V * (1.0 - w) / (n * D), prop["j_max"])
        c = prop["kq"] + self.delta
        kq = c[:, 0] + J * (c[:, 1] + J * (c[:, 2] + J * c[:, 3]))
        return 2.0 * math.pi * RHO * n ** 3 * D ** 5 * kq / 1000.0


class ODEFunc(nn.Module):
    """dh/dt = f(h): the latent dynamics integrated between consecutive observations."""

    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, h):
        return self.net(h)


def rk4_step(f, h, dt):
    """One classical Runge-Kutta step of dh/dt = f(h) over a per-sample interval dt (B, 1)."""
    k1 = f(h)
    k2 = f(h + 0.5 * dt * k1)
    k3 = f(h + 0.5 * dt * k2)
    k4 = f(h + dt * k3)
    return h + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class WindowEncoder(nn.Module):
    """Summarise the calm-water window into one latent vector.

    mode='ode'    : ODE-RNN. The latent state evolves by a Neural ODE over the real time
                    elapsed between rows and is corrected by a GRU cell at each row.
    mode='gru'    : the same recurrent update without the ODE between rows.
    mode='static' : no memory; only the latest row is used.
    """

    def __init__(self, n_in, hidden=64, mode="ode", dropout=0.1):
        super().__init__()
        self.mode, self.hidden = mode, hidden
        self.substeps = 1   # RK4 steps per interval; raised when intervals are long
        if mode == "static":
            self.net = mlp([n_in, hidden, hidden], dropout=dropout)
        else:
            self.cell = nn.GRUCell(n_in, hidden)
            if mode == "ode":
                self.f = ODEFunc(hidden)

    def forward(self, x, dt):
        if self.mode == "static":
            return torch.tanh(self.net(x[:, -1]))
        h = x.new_zeros(x.shape[0], self.hidden)
        for k in range(x.shape[1]):
            if k > 0 and self.mode == "ode":
                # Time is measured in units of the nominal logging period.
                step = (dt[:, k] / NOMINAL_DT_S).clamp(max=60.0).unsqueeze(1) / self.substeps
                for _ in range(self.substeps):
                    h = rk4_step(self.f, h, step)
            h = self.cell(x[:, k], h)
        return h


class PINODE(nn.Module):
    """Wake network + propeller layer.

    encoder : 'ode' | 'gru' | 'static' (see WindowEncoder), or 'relax': the wake itself is the
              state of a first-order ODE, dw/dt = (w_eq(x) - w) / tau, relaxing towards the
              equilibrium wake of the current operating point with one learned time constant.
    inputs  : 'raw' (physical units of the source vessel) or 'nd' (non-dimensional).
    """

    def __init__(self, vessel, hidden=64, encoder="ode", use_weather=True,
                 trainable_kq=False, dropout=0.1, inputs="raw"):
        super().__init__()
        self.calm_idx, self.weather_idx = INPUT_SETS[inputs]
        self.mode = encoder
        self.encoder = WindowEncoder(len(self.calm_idx), hidden, "static" if encoder == "relax" else encoder, dropout)
        self.decoder = mlp([hidden, 64, 32, 1], dropout=dropout)
        self.weather = mlp([len(self.weather_idx), 32, 16, 1], dropout=dropout) if use_weather else None
        self.propeller = PropellerLayer(vessel, trainable=trainable_kq)
        # Time constant of the relaxation ODE, in units of the logging period (softplus-positive).
        self.log_tau = nn.Parameter(torch.tensor(1.0))
        # Per-vessel shift of the wake logit; zero on the source vessel, the only quantity
        # re-estimated in parameter-efficient few-shot adaptation.
        self.wake_offset = nn.Parameter(torch.zeros(()))

    def _row_logit(self, x_row):
        h = self.encoder(x_row[:, None, self.calm_idx], None)
        logit = self.decoder(h).squeeze(1)
        if self.weather is not None:
            logit = logit + self.weather(x_row[:, self.weather_idx]).squeeze(1)
        return logit

    def wake(self, batch):
        x = batch["x"]
        if self.mode == "relax":
            B, L, F = x.shape
            eq = self._row_logit(x.reshape(B * L, F)).reshape(B, L)
            tau = nn.functional.softplus(self.log_tau) + 1e-3
            state = eq[:, 0]
            for k in range(1, L):
                # Exact solution of the linear ODE over the elapsed time, target held constant.
                decay = torch.exp(-(batch["dt"][:, k] / NOMINAL_DT_S).clamp(max=60.0) / tau)
                state = eq[:, k] + (state - eq[:, k]) * decay
            logit = state
        else:
            logit = self.decoder(self.encoder(x[:, :, self.calm_idx], batch["dt"])).squeeze(1)
            if self.weather is not None:
                logit = logit + self.weather(x[:, -1, self.weather_idx]).squeeze(1)
        return W_MIN + (W_MAX - W_MIN) * torch.sigmoid(logit + self.wake_offset)

    def forward(self, batch):
        prop = {k: batch[k] for k in ("D", "kq", "j_max")} if "kq" in batch else None
        return self.propeller(self.wake(batch), batch["n"], batch["V"], prop)

    def regulariser(self, batch, power):
        # Only active in the trainable-K_Q ablation: keep the cubic near the B-series.
        return 10.0 * (self.propeller.delta ** 2).sum()


class BlackBox(nn.Module):
    """Unconstrained regressors. backbone: 'mlp' (latest row), 'gru' (window), 'kan' (latest row)."""

    def __init__(self, scaler, backbone="mlp", hidden=(128, 64, 32), inputs="raw", dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.y_mean, self.y_std = scaler.y_mean, scaler.y_std
        self.idx = INPUT_SETS[inputs][0] + INPUT_SETS[inputs][1]
        F = len(self.idx)
        if backbone == "mlp":
            self.net = mlp([F, *hidden, 1], dropout=dropout)
        elif backbone == "kan":
            self.net = KAN([F, 64, 32, 1])
        elif backbone == "gru":
            self.rnn = nn.GRU(F, hidden[0] // 2, batch_first=True)
            self.net = mlp([hidden[0] // 2, 64, 32, 1], dropout=dropout)
        else:
            raise ValueError(backbone)

    def scaled(self, x):
        x = x[..., self.idx]
        if self.backbone == "gru":
            return self.net(self.rnn(x)[0][:, -1]).squeeze(1)
        return self.net(x[:, -1]).squeeze(1)

    def forward(self, batch):
        return self.scaled(batch["x"]) * self.y_std + self.y_mean


class Hybrid(BlackBox):
    """Black-box regressor with soft hull-side physics added to the loss.

    The penalty pulls the prediction towards a calm-water speed-power curve
    P_hull = k * C_F(Re) * V^3, i.e. the cubic law with the ITTC-57 friction line. The single
    constant k (wetted surface, form factor, propulsive efficiency) is calibrated on the source
    training data, so the physics target is unbiased on average and needs no hull particulars.
    At inference the model is an ordinary network; the physics only shapes training.
    """

    def __init__(self, scaler, backbone="mlp", beta=0.1, length=225.0, **kw):
        super().__init__(scaler, backbone, **kw)
        self.beta, self.length, self.k = beta, length, 1.0

    def hull_power(self, V):
        re = (V * self.length / 1.19e-6).clamp(min=1e5)
        return self.k * 0.075 / (torch.log10(re) - 2.0) ** 2 * V ** 3

    def calibrate(self, V, power):
        self.k = 1.0
        base = self.hull_power(V)
        self.k = float((base * power).sum() / (base * base).sum())

    def regulariser(self, batch, power):
        return self.beta * ((power - self.hull_power(batch["V"])) / self.y_std).pow(2).mean()


def build(kind, source, scaler, **kw):
    """kind: pinode | data | data_gru | hybrid | pikan."""
    if kind == "pinode":
        return PINODE(source, **kw)
    if kind == "data":
        return BlackBox(scaler, "mlp", **kw)
    if kind == "data_gru":
        return BlackBox(scaler, "gru", **kw)
    if kind == "hybrid":
        return Hybrid(scaler, "mlp", **kw)
    if kind == "pikan":
        return Hybrid(scaler, "kan", **kw)
    raise ValueError(kind)
