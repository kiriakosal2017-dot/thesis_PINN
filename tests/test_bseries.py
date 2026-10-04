import numpy as np

from pinode.bseries import (KQ_TERMS, KT_TERMS, open_water_cubics, polyval_ascending,
                     solve_j_from_kt)


def test_term_counts():
    assert len(KT_TERMS) == 39
    assert len(KQ_TERMS) == 47


def test_bollard_kt_matches_published_values():
    # Published checks of the Oosterveld-van Oossanen polynomial at J = 0, P/D = 1.
    kt_b470, _ = open_water_cubics(1.0, 0.70, 4)
    kt_b350, _ = open_water_cubics(1.0, 0.50, 3)
    assert abs(kt_b470[0] - 0.4547) < 5e-4
    assert abs(kt_b350[0] - 0.4057) < 5e-4


def test_open_water_curves_are_physical():
    kt, kq = open_water_cubics(1.0, 0.70, 4)
    j = np.linspace(0.05, 1.0, 40)
    kt_v, kq_v = polyval_ascending(kt, j), polyval_ascending(kq, j)
    assert np.all(np.diff(kt_v) < 0) and np.all(np.diff(kq_v) < 0)
    assert np.all(kq_v > 0)
    eta0 = j * kt_v / (2 * np.pi * kq_v)
    assert 0.55 < eta0.max() < 0.75


def test_thrust_identity_roundtrip():
    kt, _ = open_water_cubics(0.75, 0.55, 5)
    j_true = np.array([0.2, 0.45, 0.6])
    j = solve_j_from_kt(kt, polyval_ascending(kt, j_true))
    assert np.allclose(j, j_true, atol=1e-6)
    assert np.isnan(solve_j_from_kt(kt, np.array([5.0]))[0])
