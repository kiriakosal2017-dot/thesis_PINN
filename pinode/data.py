"""Vessel data for the torque-identity PI-NODE: one feature schema shared by every vessel.

Each vessel is described by a local ``.env.<name>`` file (not version-controlled) holding
the data path and the propeller particulars. All models, physics-informed or not, see the
same eleven inputs, each of which is logged on every vessel, so cross-vessel transfer never
has to invent a missing sensor. Scaling and imputation statistics always come from the
source vessel's training period.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

KNOTS_TO_MS = 0.51444
RHO = 1025.0
TARGET = "Shaft Power_TRQM"
RPM = "Propeller-Shaft-RPM"

# Model inputs. The first block describes the calm-water operating point, the second the wind.
CALM_FEATURES = ["stw", "sog", "rpm", "draft_fore", "draft_aft", "draft_mid", "trim"]
WEATHER_FEATURES = ["wind_speed", "wind_cos", "wind_sin", "wind_head"]
FEATURES = CALM_FEATURES + WEATHER_FEATURES

# The same information made non-dimensional with the vessel's own scales, so that a value
# means the same thing on every ship: apparent advance ratio J0 = V/(nD); speed, shaft speed
# and draft relative to the vessel's usual upper values; wind relative to ship speed. The
# reference values are 95th percentiles of the vessel's own inputs (no power measurement is
# involved), taken over the first 64 % of its record.
ND_CALM_FEATURES = ["nd_j0", "nd_speed", "nd_shaft", "nd_draft", "nd_trim", "nd_current"]
ND_WEATHER_FEATURES = ["nd_wind", "nd_wind_head", "nd_wind_cross"]
ND_FEATURES = ND_CALM_FEATURES + ND_WEATHER_FEATURES
ALL_FEATURES = FEATURES + ND_FEATURES

MIN_POWER_KW = 1000.0
MIN_SPEED_KN = 4.0
NOMINAL_DT_S = 60.0
MIN_DRAFT_M, MAX_DRAFT_M = 3.0, 20.0
MAX_GAP_S = 3600.0


@dataclass
class Vessel:
    name: str
    file_path: str
    D: float
    p_d: float
    ae_a0: float
    z: int
    speed_column: str


def load_vessel(name):
    env = dotenv_values(f".env.{name}")
    if not env:
        raise FileNotFoundError(f".env.{name} not found or empty")
    return Vessel(
        name=name,
        file_path=env["DATA_FILE_PATH"],
        D=float(env["PROPELLER_DIAMETER"]),
        p_d=float(env["PROPELLER_PITCH_RATIO"]),
        ae_a0=float(env["PROPELLER_AREA_RATIO"]),
        z=int(env["PROPELLER_BLADES"]),
        speed_column=env.get("SPEED_COLUMN", "Speed-Through-Water"),
    )


def _read_raw(vessel):
    cache = Path("cache") / f"{vessel.name}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    path = Path(vessel.file_path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    cache.parent.mkdir(exist_ok=True)
    df.to_pickle(cache)
    return df


def _col(df, name):
    return df[name].astype(float) if name in df.columns else pd.Series(np.nan, index=df.index)


def _draft(df, name):
    """Draft gauge reading, with physically implausible values treated as missing.

    Several vessels carry a dead or mis-scaled gauge (a fore draft of exactly zero, for
    instance). Readings outside the plausible range for ships of this size are dropped here
    and later imputed like any other missing value.
    """
    d = _col(df, name)
    return d.where((d >= MIN_DRAFT_M) & (d <= MAX_DRAFT_M))


def load_frame(vessel):
    """Return the filtered, feature-engineered frame of one vessel in logging order.

    Columns: FEATURES, plus ``power`` (kW), ``n`` (rev/s), ``V`` (m/s), ``dt`` (s) and
    ``accel`` (m/s^2). Feature NaNs are left in place; ``Scaler`` imputes them.
    """
    raw = _read_raw(vessel)
    time = pd.to_datetime(raw["TIME"], errors="coerce") if "TIME" in raw.columns else pd.Series(pd.NaT, index=raw.index)
    # Some exports carry incomplete timestamps; only re-sort when every row has one.
    if time.notna().all() and not time.is_monotonic_increasing:
        order = np.argsort(time.values, kind="stable")
        raw, time = raw.iloc[order], time.iloc[order]

    speed = _col(raw, vessel.speed_column)
    power, rpm = _col(raw, TARGET), _col(raw, RPM)
    keep = (power >= MIN_POWER_KW) & (speed >= MIN_SPEED_KN) & (rpm > 10.0)
    raw, time, speed, power, rpm = raw[keep], time[keep], speed[keep], power[keep], rpm[keep]

    mid = pd.concat([_draft(raw, "Middle draft(P)_AMS"), _draft(raw, "Middle draft(S)_AMS")], axis=1).mean(axis=1)
    fore = _draft(raw, "Fore draft_AMS").fillna(mid)
    aft = _draft(raw, "Aft draft_AMS").fillna(mid)
    wind = _col(raw, "Rel-Wind-Speed")
    angle = np.deg2rad(_col(raw, "Wind_angle_BRG_WIND"))

    out = pd.DataFrame({
        "stw": speed,
        "sog": _col(raw, "Speed-Over-Ground").fillna(speed),
        "rpm": rpm,
        "draft_fore": fore,
        "draft_aft": aft,
        "draft_mid": mid.fillna((fore + aft) / 2),
        "trim": aft - fore,
        "wind_speed": wind,
        "wind_cos": np.cos(angle),
        "wind_sin": np.sin(angle),
        "wind_head": wind * np.cos(angle),
        "power": power,
        "n": rpm / 60.0,
        "V": speed * KNOTS_TO_MS,
    }).reset_index(drop=True)

    ref = out.iloc[: int(len(out) * 0.64)]
    draft = (out["draft_fore"] + out["draft_aft"]) / 2
    draft_ref = ((ref["draft_fore"] + ref["draft_aft"]) / 2).quantile(0.95)
    wind_rel = out["wind_speed"] / out["V"]
    out["nd_j0"] = out["V"] / (out["n"] * vessel.D)
    out["nd_speed"] = out["V"] / ref["V"].quantile(0.95)
    out["nd_shaft"] = out["n"] / ref["n"].quantile(0.95)
    out["nd_draft"] = draft / draft_ref
    out["nd_trim"] = out["trim"] / draft_ref
    out["nd_current"] = (out["stw"] - out["sog"]) / out["stw"]
    out["nd_wind"] = wind_rel
    out["nd_wind_head"] = wind_rel * out["wind_cos"]
    out["nd_wind_cross"] = wind_rel * out["wind_sin"]

    dt = time.diff().dt.total_seconds().reset_index(drop=True)
    # Missing or zero intervals fall back to the nominal one-minute logging period.
    dt = dt.where((dt > 0) & dt.notna(), NOMINAL_DT_S)
    out["dt"] = dt.values
    out["accel"] = out["V"].diff().fillna(0.0) / np.minimum(out["dt"], MAX_GAP_S)
    return out


class Scaler:
    """Median imputation and standardisation fitted on the source training rows only."""

    def fit(self, frame):
        x = frame[ALL_FEATURES]
        self.median = x.median()
        filled = x.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std().replace(0.0, 1.0)
        self.y_mean = float(frame["power"].mean())
        self.y_std = float(frame["power"].std())
        return self

    def transform(self, frame):
        x = frame[ALL_FEATURES].fillna(self.median)
        return ((x - self.mean) / self.std).values.astype(np.float32)


@dataclass
class Split:
    """Rows of one vessel plus the window end-indices that make up a data split."""
    X: np.ndarray        # (N, F) scaled features, columns ordered as ALL_FEATURES
    n: np.ndarray        # (N,) shaft speed, rev/s
    V: np.ndarray        # (N,) speed through water, m/s
    y: np.ndarray        # (N,) shaft power, kW
    dt: np.ndarray       # (N,) seconds since the previous row
    accel: np.ndarray    # (N,) m/s^2
    frame: pd.DataFrame  # unscaled features, for the closed-form baselines
    ends: np.ndarray     # window end indices (target is the power at this row)


def _window_ends(dt, lo, hi, seq_len):
    """End indices in [lo, hi) of windows lying wholly inside [lo, hi) without a logging gap."""
    gap = (dt > MAX_GAP_S).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(gap)])
    ends = np.arange(lo + seq_len - 1, hi)
    # The first row's dt belongs to the interval before the window, so it is not counted.
    inside = csum[ends + 1] - csum[ends - seq_len + 2]
    return ends[inside == 0]


def make_splits(frame, scaler, fractions, seq_len):
    """Cut a vessel chronologically; ``fractions`` are cumulative boundaries, e.g. (0.64, 0.8)."""
    N = len(frame)
    X = scaler.transform(frame)
    bounds = [0] + [int(N * f) for f in fractions] + [N]
    dt = frame["dt"].values
    return [
        Split(X=X, n=frame["n"].values.astype(np.float32), V=frame["V"].values.astype(np.float32),
              y=frame["power"].values.astype(np.float32), dt=dt.astype(np.float32),
              accel=frame["accel"].values, frame=frame, ends=_window_ends(dt, lo, hi, seq_len))
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]


VAL_BLOCK_ROWS = 1440   # about one day of one-minute records
VAL_EVERY = 5


def load_source(name="source", seq_len=10):
    """Source vessel: the last 20 % of the record is the test period.

    Validation is every fifth day-long block of the first 80 %, not its final stretch. One
    voyage lasts weeks and each has its own loading and weather, so a validation period taken
    from the end of the training record is a single voyage and says little about over-fitting;
    blocks spread through the record cover the same conditions as the training rows. Windows
    never straddle a block boundary, and the test period stays strictly in the future.
    """
    vessel = load_vessel(name)
    frame = load_frame(vessel)
    N = len(frame)
    cut = int(N * 0.80)
    edges = list(range(0, cut, VAL_BLOCK_ROWS)) + [cut]
    blocks = list(zip(edges[:-1], edges[1:]))
    val_blocks = [b for i, b in enumerate(blocks) if i % VAL_EVERY == 2]
    train_blocks = [b for i, b in enumerate(blocks) if i % VAL_EVERY != 2]
    scaler = Scaler().fit(frame.iloc[np.concatenate([np.arange(*b) for b in train_blocks])])
    X, dt = scaler.transform(frame), frame["dt"].values

    def split(block_list):
        ends = np.concatenate([_window_ends(dt, lo, hi, seq_len) for lo, hi in block_list if hi - lo >= seq_len])
        return Split(X=X, n=frame["n"].values.astype(np.float32), V=frame["V"].values.astype(np.float32),
                     y=frame["power"].values.astype(np.float32), dt=dt.astype(np.float32),
                     accel=frame["accel"].values, frame=frame, ends=ends)

    return vessel, scaler, split(train_blocks), split(val_blocks), split([(cut, N)])


def load_target(name, scaler, seq_len=10):
    """Target vessel: first 80 % is the few-shot pool, last 20 % the test period."""
    vessel = load_vessel(name)
    frame = load_frame(vessel)
    pool, test = make_splits(frame, scaler, (0.80,), seq_len)
    return vessel, pool, test
