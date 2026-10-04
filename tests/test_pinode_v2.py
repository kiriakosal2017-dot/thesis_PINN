import numpy as np
import torch

from pinode.data import ALL_FEATURES, Vessel, _window_ends
from pinode.models import PINODE, PropellerLayer, W_MAX, W_MIN, rk4_step

SHIP = Vessel("t", "", D=7.0, p_d=0.75, ae_a0=0.55, z=5, speed_column="Speed-Through-Water")
OTHER = Vessel("o", "", D=6.5, p_d=0.70, ae_a0=0.48, z=4, speed_column="Speed-Through-Water")


def batch(B=6, L=10):
    return {"x": torch.randn(B, L, len(ALL_FEATURES)), "dt": torch.full((B, L), 60.0),
            "n": torch.full((B,), 1.35), "V": torch.full((B,), 6.5)}


def test_window_ends_skip_gaps_and_split_borders():
    dt = np.full(40, 60.0)
    dt[20] = 7200.0                      # logging gap just before row 20
    ends = _window_ends(dt, 5, 40, seq_len=10)
    assert ends.min() == 14              # first window lies wholly inside [5, 40)
    assert not np.any((ends >= 20) & (ends < 29))   # windows straddling the gap are dropped
    assert 19 in ends and 29 in ends


def test_power_increases_with_wake_and_shaft_speed():
    layer = PropellerLayer(SHIP)
    n, V = torch.full((5,), 1.35), torch.full((5,), 6.5)
    p = layer(torch.linspace(0.1, 0.6, 5), n, V)
    assert torch.all(p[1:] > p[:-1]) and torch.all(p > 0)
    assert layer(torch.full((5,), 0.4), n * 1.1, V).mean() > layer(torch.full((5,), 0.4), n, V).mean()


def test_swapping_the_propeller_changes_power_but_not_wake():
    torch.manual_seed(0)
    model = PINODE(SHIP).eval()
    b = batch()
    w0, p0 = model.wake(b), model(b)
    model.propeller.set_propeller(OTHER)
    assert torch.allclose(model.wake(b), w0)
    assert not torch.allclose(model(b), p0)
    assert torch.all((w0 >= W_MIN) & (w0 <= W_MAX))


def test_fixed_kq_has_no_trainable_parameters():
    assert not PINODE(SHIP).propeller.delta.requires_grad
    assert PINODE(SHIP, trainable_kq=True).propeller.delta.requires_grad


def test_ode_encoder_depends_on_elapsed_time_gru_does_not():
    torch.manual_seed(0)
    b = batch()
    slow = {**b, "dt": b["dt"] * 5}
    ode, gru = PINODE(SHIP, encoder="ode").eval(), PINODE(SHIP, encoder="gru").eval()
    assert not torch.allclose(ode.wake(b), ode.wake(slow))
    assert torch.allclose(gru.wake(b), gru.wake(slow))


def test_rk4_integrates_linear_decay():
    h = rk4_step(lambda h: -h, torch.ones(3, 1), torch.full((3, 1), 0.1))
    assert torch.allclose(h, torch.full((3, 1), float(np.exp(-0.1))), atol=1e-6)


def test_relaxation_encoder_follows_elapsed_time():
    torch.manual_seed(0)
    model = PINODE(SHIP, encoder="relax", inputs="nd").eval()
    b = batch()
    # After a very long interval the state has fully relaxed to the latest row's equilibrium.
    late = {**b, "dt": b["dt"] * 1000}
    last_only = {**b, "x": b["x"][:, -1:].repeat(1, b["x"].shape[1], 1)}
    assert torch.allclose(model.wake(late), model.wake(last_only), atol=1e-5)
    assert not torch.allclose(model.wake(b), model.wake(late))


def test_wake_offset_shifts_the_wake():
    model = PINODE(SHIP, encoder="static").eval()
    b = batch()
    w0 = model.wake(b)
    with torch.no_grad():
        model.wake_offset.fill_(0.5)
    assert torch.all(model.wake(b) > w0)


def test_sparse_windows_keep_order_target_row_and_elapsed_time():
    from pinode.data import Split
    from pinode.training import Windows
    N = 200
    dt = np.full(N, 60.0, dtype=np.float32)
    split = Split(X=np.arange(N, dtype=np.float32)[:, None].repeat(len(ALL_FEATURES), 1), n=np.ones(N, np.float32),
                  V=np.ones(N, np.float32), y=np.ones(N, np.float32), dt=dt, accel=np.zeros(N), frame=None,
                  ends=np.arange(100, 150))
    w = Windows(split, 10, span=30)
    b, _ = w.batch(w.ends)
    rows = b["x"][:, :, 0]
    assert torch.all(rows[:, -1] == w.ends) and torch.all(rows[:, 1:] > rows[:, :-1])
    assert torch.all(rows[:, 0] >= w.ends - 29)
    assert torch.allclose(b["dt"][:, 1:], (rows[:, 1:] - rows[:, :-1]) * 60.0)
    w.reseed()
    assert torch.equal(w.batch(w.ends)[0]["x"], b["x"])


def test_per_sample_propellers_match_single_vessel_layer():
    from pinode.bseries import open_water_cubics
    layer_a, layer_b = PropellerLayer(SHIP), PropellerLayer(OTHER)
    w, n, V = torch.full((4,), 0.4), torch.full((4,), 1.3), torch.full((4,), 6.5)
    prop = {"D": torch.tensor([SHIP.D, OTHER.D, SHIP.D, OTHER.D]),
            "kq": torch.tensor(np.stack([open_water_cubics(v.p_d, v.ae_a0, v.z)[1] for v in (SHIP, OTHER, SHIP, OTHER)]), dtype=torch.float32),
            "j_max": torch.stack([layer_a.j_max, layer_b.j_max, layer_a.j_max, layer_b.j_max])}
    mixed = layer_a(w, n, V, prop)
    assert torch.allclose(mixed[0::2], layer_a(w, n, V)[0::2], rtol=1e-5)
    assert torch.allclose(mixed[1::2], layer_b(w, n, V)[1::2], rtol=1e-5)
