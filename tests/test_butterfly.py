"""Reproduce Table 2 (Appendix C) of Deschâtres, SSRN 4831218: ETH, 2023-12-15 expiry."""

import numpy as np
import pytest

from cvi.fitter import butterfly as bf

T, SIGMA_STAR = 0.0265, 0.4853
V_STAR = SIGMA_STAR**2

# z, v_ref, s_ref, beta0, beta1, beta2 as printed in the paper
TABLE2_INTERIOR = [
    (-6.0, 1.21501, -1.31035, 0.0543, -0.80983, -0.73423),
    (-4.0, 0.70172, -0.86944, -0.21948, -1.41118, -1.56605),
    (-2.0, 0.37537, -0.56040, -0.74223, -1.97955, -2.69160),
    (-1.0, 0.26557, -0.32237, -1.42302, -1.80625, -2.01845),
    (0.0, 0.23528, 0.06511, -1.99788, 0.06527, -0.00902),
    (1.0, 0.28751, 0.34138, -1.43207, 1.68919, -1.83909),
    (2.0, 0.38733, 0.50644, -0.87986, 1.99175, -2.40194),
    (4.0, 0.70358, 0.83656, -0.26959, 1.45946, -1.56731),
    (6.0, 1.16621, 1.10877, -0.09056, 1.02123, -0.86286),
]


@pytest.mark.parametrize("z, v, s, b0, b1, b2", TABLE2_INTERIOR)
def test_interior_linearization_matches_table2(z, v, s, b0, b1, b2):
    k = z * SIGMA_STAR * np.sqrt(T)
    c0, c1, c2 = bf.linearized_interior(
        np.array([v]), np.array([s]), np.array([k]), V_STAR, T
    )
    # the table's inputs are rounded to 5 decimals
    assert c0[0] == pytest.approx(b0, abs=5e-4)
    assert c1[0] == pytest.approx(b1, abs=5e-4)
    assert c2[0] == pytest.approx(b2, abs=5e-4)


@pytest.mark.parametrize(
    "z, v, side, bound, slope",
    [
        (-8.0, 1.90134, "left", -1.48709, -0.67787),
        (8.0, 1.72203, "right", 1.3644, 0.69068),
    ],
)
def test_edge_bounds_match_table2(z, v, side, bound, slope):
    k = z * SIGMA_STAR * np.sqrt(T)
    assert bf.d1d2(v, k, T) > 1.0
    b, sl = bf.linearized_edge(v, k, V_STAR, T, side)
    assert b == pytest.approx(bound, abs=5e-4)
    assert sl == pytest.approx(slope, abs=5e-4)


def test_edge_bound_is_root_of_g_at_zero_convexity():
    for z, v, side in [(-8.0, 1.90134, "left"), (8.0, 1.72203, "right")]:
        k = z * SIGMA_STAR * np.sqrt(T)
        s = bf.edge_skew_bound(v, k, V_STAR, T, side)
        assert bf.g(np.array([v]), np.array([s]), np.array([k]), V_STAR, T)[
            0
        ] == pytest.approx(0.0, abs=1e-10)


def test_analytic_derivatives_match_finite_differences():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.02, 0.5, 50)
    s = rng.uniform(-2, 2, 50)
    k = rng.uniform(-1, 1, 50)
    h = 1e-6
    fd_s = (bf.g(v, s + h, k, 0.04, 0.5) - bf.g(v, s - h, k, 0.04, 0.5)) / (2 * h)
    fd_v = (bf.g(v + h, s, k, 0.04, 0.5) - bf.g(v - h, s, k, 0.04, 0.5)) / (2 * h)
    np.testing.assert_allclose(bf.dg_ds(v, s, k, 0.04, 0.5), fd_s, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(bf.dg_dv(v, s, k, 0.04, 0.5), fd_v, rtol=1e-5, atol=1e-5)
