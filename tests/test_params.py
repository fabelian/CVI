import numpy as np
import pytest
from scipy.interpolate import splev

from cvi.slice import CviCubicBSplineParams, CviNode, CviRealParams, CviSlice
from cvi.slice.cubic_spline import CubicSpline

NODES = [
    CviNode(-6.0, 0.0),
    CviNode(-4.0, 0.03),
    CviNode(-2.0, 0.06),
    CviNode(0.0, 0.12),
    CviNode(2.0, 0.03),
    CviNode(4.0, 0.015),
    CviNode(6.0, 0.0),
]


def real_params():
    return CviRealParams(atm_var=0.04, skew=5.0, nodes=NODES)


def test_dual_parameterization_matches_cubic_spline():
    """B-spline weights (Section 2.2) and the cubic spline (Section 2.1) agree."""
    p = real_params()
    spline = CviCubicBSplineParams.from_real_params(p)
    cubic = CubicSpline.cvi_spline_params(
        p.atm_var, p.skew, np.array(p.locs), np.array(p.crvs)
    )
    z = np.linspace(-5.99, 5.99, 301)
    v_b = splev(z, spline.tck)
    v_c, dv_c, d2v_c = cubic(z)
    np.testing.assert_allclose(v_b, v_c, atol=1e-12)
    np.testing.assert_allclose(splev(z, spline.tck, der=1), dv_c, atol=1e-12)
    np.testing.assert_allclose(splev(z, spline.tck, der=2), d2v_c, atol=1e-12)


def test_physical_parameters_are_reproduced():
    p = real_params()
    spline = CviCubicBSplineParams.from_real_params(p)
    assert splev(0.0, spline.tck) == pytest.approx(0.04)
    assert splev(0.0, spline.tck, der=1) == pytest.approx(p.skew)
    np.testing.assert_allclose(splev(p.locs, spline.tck, der=2), p.crvs, atol=1e-12)


def test_round_trip_real_spline_real():
    p = real_params()
    q = CviRealParams.from_spline_params(CviCubicBSplineParams.from_real_params(p))
    assert q.atm_var == pytest.approx(p.atm_var)
    assert q.skew == pytest.approx(p.skew)
    np.testing.assert_allclose(q.crvs, p.crvs, atol=1e-12)


def test_missing_atm_node_is_rejected():
    with pytest.raises(ValueError):
        CviRealParams(0.04, 0.0, [CviNode(-1.0, 0.0), CviNode(1.0, 0.0)])


def test_strike_derivatives_match_finite_differences():
    slc = CviSlice.from_real_params(real_params(), ref_fwd=100.0, t_e=1.0)
    K = np.array([60.0, 90.0, 100.0, 115.0, 150.0])
    vol, dvol, d2vol = slc.vol_deriv1_deriv2_k(K)
    h = 1e-3
    vp = slc.vol_deriv1_deriv2_k(K + h)[0]
    vm = slc.vol_deriv1_deriv2_k(K - h)[0]
    np.testing.assert_allclose(dvol, (vp - vm) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(d2vol, (vp - 2 * vol + vm) / h**2, rtol=1e-4, atol=1e-8)


def test_linear_extrapolation_beyond_edge_knots():
    slc = CviSlice.from_real_params(real_params(), ref_fwd=100.0, t_e=1.0)
    z = np.array([6.5, 7.0, 8.0])
    v, dv, d2v = slc.var_deriv1_deriv2_z(z)
    np.testing.assert_allclose(np.diff(v) / np.diff(z), dv[0], rtol=1e-10)
    np.testing.assert_allclose(d2v, 0.0)
