import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvi.fitter import butterfly as bf
from cvi.fitter.fitter import CviVolFitter
from cvi.fitter.fitter_options import CviVolFitterOptions

DATA = Path(__file__).parents[1] / "cvi" / "examples" / "data" / "2023-04-14_vol.csv"
NODES = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]


def svi_total_variance(k, t):
    """A simple arbitrage-free SSVI surface (Gatheral-Jacquier, power-law phi)."""
    theta = 0.04 * t + 0.01 * np.sqrt(t)
    rho, eta, gamma = -0.6, 1.0, 0.4
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
    return (
        0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    )


def synthetic_chain():
    rows = []
    for t in [0.1, 0.25, 0.5, 1.0, 2.0]:
        fwd = 100.0 * np.exp(0.02 * t)
        for k in np.linspace(-0.6, 0.4, 41) * np.sqrt(t):
            vol = np.sqrt(svi_total_variance(k, t) / t)
            half = 0.003 + 0.01 * abs(k)
            rows.append(
                {
                    "expiry": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(days=round(365 * t)),
                    "strike": fwd * np.exp(k),
                    "t_e": t,
                    "fwd_mid": fwd,
                    "iv_bid": vol - half,
                    "iv_ask": vol + half,
                    "iv_mid": vol,
                    "vega_bid": 1.0,
                    "vega_ask": 1.0,
                }
            )
    return pd.DataFrame(rows)


def butterfly_margin(slc, z):
    v, dv, d2v = slc.var_deriv1_deriv2_z(z)
    vs = slc.atm_anchor_var
    return d2v / vs - bf.g(v, dv / vs, z * slc._z_denom, vs, slc._t_e)


def check_surface(result, chain, z_grid, min_inside):
    assert result.success, result.statuses
    assert result.butterfly_violations[-1] == 0
    for slc in result.slices.values():
        assert butterfly_margin(slc, z_grid).min() > -1e-6
    inside = total = 0
    for exp, slc in result.slices.items():
        df = chain[chain["expiry"] == exp]
        z = slc.k_to_z(df["strike"].to_numpy())
        ok = (
            (z >= NODES[0])
            & (z <= NODES[-1])
            & (df["iv_bid"] < df["iv_ask"]).to_numpy()
        )
        vol = np.sqrt(slc.var_deriv1_deriv2_z(z[ok])[0])
        inside += np.sum(
            (vol >= df["iv_bid"].to_numpy()[ok] - 1e-9)
            & (vol <= df["iv_ask"].to_numpy()[ok] + 1e-9)
        )
        total += ok.sum()
    assert inside / total >= min_inside


def test_synthetic_ssvi_surface_is_recovered():
    chain = synthetic_chain()
    res = CviVolFitter().fit(chain, NODES, CviVolFitterOptions())
    check_surface(res, chain, np.linspace(-6, 6, 1201), min_inside=0.99)
    # calendar: total variance increasing at fixed log-moneyness
    slices = sorted(res.slices.values(), key=lambda s: s._t_e)
    k = np.linspace(-0.5, 0.4, 500)
    for a, b in pairwise(slices):
        wa = a._t_e * a.var_deriv1_deriv2_z(k / a._z_denom)[0]
        wb = b._t_e * b.var_deriv1_deriv2_z(k / b._z_denom)[0]
        assert np.min(wb - wa) > -1e-6


@pytest.mark.skipif(not DATA.exists(), reason="example data missing")
def test_spx_example_is_arbitrage_free():
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
    chain = pd.read_csv(DATA, parse_dates=["expiry"])
    res = CviVolFitter().fit(chain, NODES, CviVolFitterOptions())
    check_surface(res, chain, np.linspace(-6, 6, 2001), min_inside=0.99)
    assert res.n_iterations <= 4  # converges in 3 solves with these knots
