"""Closed-form and classical baselines. None uses a neural network.

Every model is fitted on the source training period and exposes
``predict(split, vessel) -> power in kW``; ``vessel`` supplies the propeller particulars
when the model is applied to another ship.
"""
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from pinode.bseries import open_water_cubics, polyval_ascending
from pinode.data import FEATURES, RHO


def _nv(split):
    rows = split.ends
    return split.n[rows].astype(float), split.V[rows].astype(float), split.y[rows].astype(float)


def _base(n, D):
    return 2.0 * math.pi * RHO * n ** 3 * D ** 5 / 1000.0


def torque_identity_wake(split, vessel):
    """Effective wake at which the open-water K_Q reproduces the measured torque."""
    n, V, y = _nv(split)
    _, kq = open_water_cubics(vessel.p_d, vessel.ae_a0, vessel.z)
    grid = np.linspace(0.0, 1.2, 2401)
    vals = polyval_ascending(kq, grid)
    stop = int(np.argmax(vals <= 0)) if np.any(vals <= 0) else len(grid)
    J = np.interp(y / _base(n, vessel.D), vals[:stop][::-1], grid[:stop][::-1])
    return 1.0 - J * n * vessel.D / V


class CubicLaw:
    """P = K_Q * 2*pi*rho*n^3*D^5 with one constant K_Q (the propeller law)."""
    name = "Propeller law (constant K_Q)"

    def fit(self, train, vessel):
        n, _, y = _nv(train)
        b = _base(n, vessel.D)
        self.kq = float(b @ y / (b @ b))
        return self

    def predict(self, split, vessel):
        n, _, _ = _nv(split)
        return self.kq * _base(n, vessel.D)


class FittedKQ:
    """P = 2*pi*rho*n^3*D^5 * K_Q(J0), K_Q a quadratic in J0 = V/(nD) fitted by least squares."""
    name = "Fitted K_Q(J) quadratic"

    def _design(self, split, vessel):
        n, V, _ = _nv(split)
        J = V / (n * vessel.D)
        return np.stack([_base(n, vessel.D) * J ** k for k in range(3)], axis=1)

    def fit(self, train, vessel):
        self.c, *_ = np.linalg.lstsq(self._design(train, vessel), _nv(train)[2], rcond=None)
        return self

    def predict(self, split, vessel):
        return self._design(split, vessel) @ self.c


class BSeriesConstantWake:
    """Torque identity with the vessel's B-series K_Q and one constant wake fraction.

    This is PI-NODE with the network removed: the wake is the median torque-identity wake of
    the source training period.
    """
    name = "B-series, constant wake"

    def fit(self, train, vessel):
        self.w = float(np.median(torque_identity_wake(train, vessel)))
        return self

    def predict(self, split, vessel):
        n, V, _ = _nv(split)
        _, kq = open_water_cubics(vessel.p_d, vessel.ae_a0, vessel.z)
        grid = np.linspace(0.0, 1.2, 2401)
        vals = polyval_ascending(kq, grid)
        j_max = grid[int(np.argmax(vals <= 0)) - 1] if np.any(vals <= 0) else 1.2
        J = np.minimum(V * (1.0 - self.w) / (n * vessel.D), j_max)
        return _base(n, vessel.D) * polyval_ascending(kq, J)


class PhysicsRidge:
    """Ridge regression on cubic speed/shaft-speed monomials and their draft interactions."""
    name = "Ridge on physical monomials"

    def _design(self, split):
        n, V, _ = _nv(split)
        f = split.frame.iloc[split.ends]
        fore = f["draft_fore"].fillna(self.fill["draft_fore"]).values
        aft = f["draft_aft"].fillna(self.fill["draft_aft"]).values
        return np.column_stack([n ** 3, n ** 2 * V, n * V ** 2, V ** 3, n ** 3 * fore, n ** 3 * aft])

    def fit(self, train, vessel):
        self.fill = train.frame.iloc[train.ends][["draft_fore", "draft_aft"]].median()
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(self._design(train), _nv(train)[2])
        return self

    def predict(self, split, vessel):
        return self.model.predict(self._design(split))


class GradientBoosting:
    """Histogram gradient boosting on the same eleven scaled inputs as the neural models."""
    name = "Gradient boosting"

    def fit(self, train, vessel):
        self.model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        self.model.fit(train.X[train.ends][:, :len(FEATURES)], _nv(train)[2])
        return self

    def predict(self, split, vessel):
        return self.model.predict(split.X[split.ends][:, :len(FEATURES)])


ALL = [CubicLaw, FittedKQ, BSeriesConstantWake, PhysicsRidge, GradientBoosting]
assert len(FEATURES) == 11
