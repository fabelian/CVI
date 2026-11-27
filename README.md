# CVI

Implementation of **Convex Volatility Interpolation** — Fabrice Deschâtres,
[SSRN 4831218](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4831218)
(Risk, Cutting Edge, February 2026).

The implied variance of each expiry is a cubic spline in the normalized
log-moneyness `z = log(K/F) / (sigma_star sqrt(T))`, with two equivalent
parameterizations linked by a linear map:

- **physical** (`CviRealParams`): ATM variance, ATM skew and the convexity at each knot;
- **B-spline** (`CviCubicBSplineParams`): the weights of the cubic B-spline basis.

The whole surface is calibrated by a sequence of convex QPs solved directly
with [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) (no modelling layer):

| Paper | Implementation (`cvi/fitter/fitter.py`) |
|---|---|
| Least squares on mid variance, weights `1/(v_ask - v_bid)^2`, scaled per expiry | ✅ |
| Above-ask / below-bid penalties, vega weights normalized by `q_j` | ✅ (one-sided quotes by default, `bid_ask_penalty_scope="all"` for every quote) |
| Strike regularization `lambda * sum |c_i - c_{i+1}|` (lambda = 0.05) | ✅ (auxiliary variables) |
| Positivity of the variance (first expiry) | ✅ |
| Zero convexity at edge knots, increasing wings (eq. 1) | ✅ |
| Lee's tail slope bounds (eq. 2) | ✅ |
| No calendar spread at fixed `K/F` (eq. 3), in the tails (eq. 4) | ✅ (linear extrapolation outside the knots) |
| Linearized no-butterfly constraints inside (eq. 5) and at the edge knots (eqs. 18, 19) | ✅ (`cvi/fitter/butterfly.py`) |
| First solve without butterfly constraints, then re-linearize until arbitrage-free | ✅ |

## Usage

```python
import pandas as pd
from cvi.fitter.fitter import CviVolFitter
from cvi.fitter.fitter_options import CviVolFitterOptions

chain = pd.read_csv("cvi/examples/data/2023-04-14_vol.csv", parse_dates=["expiry"])
nodes = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
result = CviVolFitter().fit(chain, nodes, CviVolFitterOptions())
slc = result.slices[pd.Timestamp("2023-06-16 16:00")]
vol, dvol_dk, d2vol_dk2 = slc.vol_deriv1_deriv2_k(4100.0)
```

`python -m cvi.examples.fit_spx [--plot]` fits the bundled S&P 500 snapshot
(54 expiries, about 9,000 strikes): four Clarabel solves, about 0.8 s of solver
time, no butterfly arbitrage on a 2,001-point grid per expiry, and 99.5% of
the two-sided quotes inside the bid-ask.

## Notes

- Constraints are imposed on finite grids (`n_*_points` options), as in the paper;
  tiny violations between grid points are possible. The butterfly grid always
  includes the interior knots and a denser set in the two edge segments.
- Quotes whose bid vol is above the ask vol (from mixing call and put quotes)
  are ignored; expiries shorter than `min_t_e` are dropped.
- The trader-facing skew of `CviRealParams(atm_var, skew, nodes)` is
  `-100 * dv/dz`; it is stored internally as `dv/dz`.

## Tests

```bash
pytest
```

`tests/test_butterfly.py` reproduces Table 2 (Appendix C) of the paper.
