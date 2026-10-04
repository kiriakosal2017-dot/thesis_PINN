"""Wageningen B-series open-water characteristics.

Regression polynomials of Oosterveld and van Oossanen (1975), "Further computer-analyzed
data of the Wageningen B-screw series", Int. Shipbuild. Prog. 22(251), valid at Rn = 2e6:

    K_T = sum_i C_i * J^s_i * (P/D)^t_i * (A_E/A_0)^u_i * Z^v_i      (39 terms)
    K_Q = sum_i C_i * J^s_i * (P/D)^t_i * (A_E/A_0)^u_i * Z^v_i      (47 terms)

For a fixed propeller (P/D, A_E/A_0, Z) both collapse to cubics in J, which is the form the
PI-NODE physics layer consumes: ``open_water_cubics`` returns the four K_T and four K_Q
coefficients in ascending powers of J.
"""
import numpy as np

# (C, s, t, u, v): exponents of J, P/D, A_E/A_0, Z.
KT_TERMS = [
    (0.00880496, 0, 0, 0, 0), (-0.204554, 1, 0, 0, 0), (0.166351, 0, 1, 0, 0),
    (0.158114, 0, 2, 0, 0), (-0.147581, 2, 0, 1, 0), (-0.481497, 1, 1, 1, 0),
    (0.415437, 0, 2, 1, 0), (0.0144043, 0, 0, 0, 1), (-0.0530054, 2, 0, 0, 1),
    (0.0143481, 0, 1, 0, 1), (0.0606826, 1, 1, 0, 1), (-0.0125894, 0, 0, 1, 1),
    (0.0109689, 1, 0, 1, 1), (-0.133698, 0, 3, 0, 0), (0.00638407, 0, 6, 0, 0),
    (-0.00132718, 2, 6, 0, 0), (0.168496, 3, 0, 1, 0), (-0.0507214, 0, 0, 2, 0),
    (0.0854559, 2, 0, 2, 0), (-0.0504475, 3, 0, 2, 0), (0.010465, 1, 6, 2, 0),
    (-0.00648272, 2, 6, 2, 0), (-0.00841728, 0, 3, 0, 1), (0.0168424, 1, 3, 0, 1),
    (-0.00102296, 3, 3, 0, 1), (-0.0317791, 0, 3, 1, 1), (0.018604, 1, 0, 2, 1),
    (-0.00410798, 0, 2, 2, 1), (-0.000606848, 0, 0, 0, 2), (-0.0049819, 1, 0, 0, 2),
    (0.0025983, 2, 0, 0, 2), (-0.000560528, 3, 0, 0, 2), (-0.00163652, 1, 2, 0, 2),
    (-0.000328787, 1, 6, 0, 2), (0.000116502, 2, 6, 0, 2), (0.000690904, 0, 0, 1, 2),
    (0.00421749, 0, 3, 1, 2), (0.0000565229, 3, 6, 1, 2), (-0.00146564, 0, 3, 2, 2),
]

KQ_TERMS = [
    (0.00379368, 0, 0, 0, 0), (0.00886523, 2, 0, 0, 0), (-0.032241, 1, 1, 0, 0),
    (0.00344778, 0, 2, 0, 0), (-0.0408811, 0, 1, 1, 0), (-0.108009, 1, 1, 1, 0),
    (-0.0885381, 2, 1, 1, 0), (0.188561, 0, 2, 1, 0), (-0.00370871, 1, 0, 0, 1),
    (0.00513696, 0, 1, 0, 1), (0.0209449, 1, 1, 0, 1), (0.00474319, 2, 1, 0, 1),
    (-0.00723408, 2, 0, 1, 1), (0.00438388, 1, 1, 1, 1), (-0.0269403, 0, 2, 1, 1),
    (0.0558082, 3, 0, 1, 0), (0.0161886, 0, 3, 1, 0), (0.00318086, 1, 3, 1, 0),
    (0.015896, 0, 0, 2, 0), (0.0471729, 1, 0, 2, 0), (0.0196283, 3, 0, 2, 0),
    (-0.0502782, 0, 1, 2, 0), (-0.030055, 3, 1, 2, 0), (0.0417122, 2, 2, 2, 0),
    (-0.0397722, 0, 3, 2, 0), (-0.00350024, 0, 6, 2, 0), (-0.0106854, 3, 0, 0, 1),
    (0.00110903, 3, 3, 0, 1), (-0.000313912, 0, 6, 0, 1), (0.0035985, 3, 0, 1, 1),
    (-0.00142121, 0, 6, 1, 1), (-0.00383637, 1, 0, 2, 1), (0.0126803, 0, 2, 2, 1),
    (-0.00318278, 2, 3, 2, 1), (0.00334268, 0, 6, 2, 1), (-0.00183491, 1, 1, 0, 2),
    (0.000112451, 3, 2, 0, 2), (-0.0000297228, 3, 6, 0, 2), (0.000269551, 1, 0, 1, 2),
    (0.00083265, 2, 0, 1, 2), (0.00155334, 0, 2, 1, 2), (0.000302683, 0, 6, 1, 2),
    (-0.0001843, 0, 0, 2, 2), (-0.000425399, 0, 3, 2, 2), (0.0000869243, 3, 3, 2, 2),
    (-0.0004659, 0, 6, 2, 2), (0.0000554194, 1, 6, 2, 2),
]


def _cubic(terms, p_d, ae_a0, z):
    coeffs = np.zeros(4)
    for c, s, t, u, v in terms:
        coeffs[s] += c * p_d ** t * ae_a0 ** u * z ** v
    return coeffs


def open_water_cubics(p_d, ae_a0, z):
    """Return (kt, kq): cubic coefficients in ascending powers of J for one propeller."""
    return _cubic(KT_TERMS, p_d, ae_a0, z), _cubic(KQ_TERMS, p_d, ae_a0, z)


def polyval_ascending(coeffs, j):
    """Evaluate c0 + c1*J + c2*J^2 + c3*J^3 (works for numpy arrays and torch tensors)."""
    return coeffs[0] + j * (coeffs[1] + j * (coeffs[2] + j * coeffs[3]))


def solve_j_from_kt(kt_coeffs, kt_target, j_lo=0.0, j_hi=None, iters=60):
    """Thrust identity: the J at which the open-water K_T equals a measured K_T.

    K_T(J) is monotonically decreasing over the working range, so bisection between
    ``j_lo`` and the zero-thrust advance ratio is sufficient. Targets outside the
    attainable range return NaN.
    """
    kt_target = np.asarray(kt_target, dtype=float)
    if j_hi is None:
        grid = np.linspace(0.0, 2.0, 2001)
        vals = polyval_ascending(kt_coeffs, grid)
        j_hi = float(grid[np.argmax(vals <= 0.0)]) if np.any(vals <= 0.0) else 2.0
    lo = np.full_like(kt_target, j_lo)
    hi = np.full_like(kt_target, j_hi)
    bad = (kt_target > polyval_ascending(kt_coeffs, lo)) | (kt_target < polyval_ascending(kt_coeffs, hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = polyval_ascending(kt_coeffs, mid) > kt_target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    j = 0.5 * (lo + hi)
    return np.where(bad, np.nan, j)
