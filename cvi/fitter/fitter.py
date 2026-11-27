"""CVI calibration as a sequence of convex QPs solved directly with Clarabel.

Implements Section 3 of Deschâtres, *Convex Volatility Interpolation*
(SSRN 4831218): least squares + above-ask + below-bid + strike regularization
objective, with positivity, boundary, Lee, calendar and linearized butterfly
constraints. The first solve has no butterfly constraints; subsequent solves
linearize them at the previous solution until no butterfly arbitrage remains.

Canonical form passed to Clarabel:
    minimize 1/2 x' P x + q' x   s.t.   A x + s = b,  s in K,
with K = ZeroCone (equalities) x NonnegativeCone (A_i x <= b_i).
"""

import time
from dataclasses import dataclass, field

import clarabel as cb
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandera.typing import DataFrame
from scipy import sparse
from scipy.interpolate import splev

from cvi.option_chain import OptionChain
from cvi.slice import CviCubicBSplineParams, CviNode, CviRealParams, CviSlice

from . import butterfly as bf
from .fitter_options import CviVolFitterOptions

FloatArr = npt.NDArray[np.float64]


@dataclass
class _Expiry:
    expiry: pd.Timestamp
    t_e: float
    fwd: float
    sigma_star: float
    z_mid: FloatArr
    v_mid: FloatArr
    w_mid: FloatArr
    z_ask: FloatArr
    v_ask: FloatArr
    w_ask: FloatArr
    z_bid: FloatArr
    v_bid: FloatArr
    w_bid: FloatArr
    n_crossed: int = 0

    @property
    def v_star(self) -> float:
        return self.sigma_star**2

    @property
    def z_to_k(self) -> float:
        return self.sigma_star * np.sqrt(self.t_e)


@dataclass
class CviFitResult:
    slices: dict[pd.Timestamp, CviSlice]
    statuses: list[str]
    solve_times: list[float]
    butterfly_violations: list[int]  # after each solve, summed over expiries
    n_iterations: int
    objective: float
    dropped_expiries: list[pd.Timestamp] = field(default_factory=list)
    n_crossed_quotes: int = 0

    @property
    def success(self) -> bool:
        return all(s == "Solved" for s in self.statuses)


class _Basis:
    """Cubic B-spline basis on the CVI knots, with linear extrapolation."""

    def __init__(self, node_locs: list[float]):
        template = CviCubicBSplineParams.from_real_params(
            CviRealParams(0.04, 0.0, [CviNode(loc, 0.0) for loc in node_locs])
        )
        self.knots = template.knots
        self.locs = np.asarray(node_locs, dtype=float)
        self.z0 = float(self.locs[0])
        self.zn = float(self.locs[-1])
        self.m = len(self.knots) - 4  # number of basis functions = n + 2

    def _raw(self, z: FloatArr, der: int) -> FloatArr:
        res = np.zeros((len(z), self.m))
        c = np.zeros(self.m)
        for i in range(self.m):
            c[:] = 0.0
            c[i] = 1.0
            res[:, i] = splev(z, (self.knots, c, 3), der=der, ext=1)
        return res

    def rows(self, z: FloatArr | float, der: int) -> FloatArr:
        """Rows mapping B-spline weights to d^der v / dz^der at z (linear tails)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        inside = np.clip(z, self.z0, self.zn)
        res = self._raw(inside, der)
        left, right = z < self.z0, z > self.zn
        if der == 2:
            res[left | right] = 0.0
        elif left.any() or right.any():
            d_left = self._raw(np.array([self.z0]), 1)[0]
            d_right = self._raw(np.array([self.zn]), 1)[0]
            if der == 0:
                res[left] += np.outer(z[left] - self.z0, d_left)
                res[right] += np.outer(z[right] - self.zn, d_right)
        return res


class _QP:
    """Accumulates P, q and constraint blocks over a flat variable vector."""

    def __init__(self, n_var: int):
        self.n_var = n_var
        self.P = sparse.csc_matrix((n_var, n_var))
        self.q = np.zeros(n_var)
        self._eq: list[tuple[sparse.csr_matrix, FloatArr]] = []
        self._ineq: list[tuple[sparse.csr_matrix, FloatArr]] = []

    def block(self, mat: FloatArr, offset: int) -> sparse.csr_matrix:
        coo = sparse.coo_matrix(np.atleast_2d(mat))
        return sparse.csr_matrix(
            (coo.data, (coo.row, coo.col + offset)),
            shape=(coo.shape[0], self.n_var),
        )

    def eq(self, a: sparse.csr_matrix, b: FloatArr | float) -> None:
        self._eq.append((a, np.atleast_1d(np.asarray(b, dtype=float))))

    def le(self, a: sparse.csr_matrix, b: FloatArr | float) -> None:
        self._ineq.append((a, np.atleast_1d(np.asarray(b, dtype=float))))

    def add_quadratic(self, a: sparse.csr_matrix, y: FloatArr, w: FloatArr) -> None:
        """Adds sum_i w_i (a_i x - y_i)^2."""
        W = sparse.diags(w)
        self.P = self.P + 2.0 * (a.T @ W @ a)
        self.q = self.q - 2.0 * (a.T @ (w * y))

    def solve(
        self, extra_ineq: list[tuple[sparse.csr_matrix, FloatArr]], verbose: bool
    ) -> tuple[str, FloatArr, float, float]:
        eq = self._eq
        ineq = self._ineq + extra_ineq
        blocks = [a for a, _ in eq] + [a for a, _ in ineq]
        A = sparse.vstack(blocks, format="csc")
        b = np.concatenate([b for _, b in eq] + [b for _, b in ineq])
        n_eq = sum(a.shape[0] for a, _ in eq)
        cones = [cb.ZeroConeT(n_eq), cb.NonnegativeConeT(A.shape[0] - n_eq)]
        P = sparse.triu(self.P, format="csc")
        settings = cb.DefaultSettings()
        settings.verbose = verbose
        start = time.perf_counter()
        sol = cb.DefaultSolver(P, self.q, A, b, cones, settings).solve()
        elapsed = time.perf_counter() - start
        return str(sol.status), np.asarray(sol.x), float(sol.obj_val), elapsed


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]

    def __init__(self):
        self._slices = {}

    # ------------------------------------------------------------------ data
    @staticmethod
    def _prepare(
        chain: DataFrame[OptionChain],
        basis: _Basis,
        opts: CviVolFitterOptions,
    ) -> tuple[list[_Expiry], list[pd.Timestamp]]:
        expiries: list[_Expiry] = []
        dropped: list[pd.Timestamp] = []
        for exp, df in chain.groupby("expiry", sort=True):
            df = df.sort_values("strike")
            t_e = float(df["t_e"].iloc[0])
            if t_e < opts.min_t_e:
                dropped.append(exp)
                continue
            fwd = float(df["fwd_mid"].iloc[0])

            iv_bid = df["iv_bid"].to_numpy(dtype=float)
            iv_ask = df["iv_ask"].to_numpy(dtype=float)
            crossed = np.isfinite(iv_bid) & np.isfinite(iv_ask) & (iv_bid >= iv_ask)
            iv_bid = np.where(crossed | (iv_bid <= 0), np.nan, iv_bid)
            iv_ask = np.where(crossed | (iv_ask <= 0), np.nan, iv_ask)
            has_bid, has_ask = np.isfinite(iv_bid), np.isfinite(iv_ask)
            has_mid = has_bid & has_ask

            strikes = df["strike"].to_numpy(dtype=float)
            if not has_mid.any():
                dropped.append(exp)
                continue
            # anchor: mid vol nearest to the forward
            idx_mid = np.flatnonzero(has_mid)
            atm = idx_mid[np.argmin(np.abs(strikes[idx_mid] - fwd))]
            sigma_star = 0.5 * (iv_bid[atm] + iv_ask[atm])
            z = np.log(strikes / fwd) / (sigma_star * np.sqrt(t_e))
            in_range = (z >= basis.z0) & (z <= basis.zn)

            v_bid, v_ask = iv_bid**2, iv_ask**2
            mid = has_mid & in_range
            if mid.sum() < 3:
                dropped.append(exp)
                continue
            spread = v_ask[mid] - v_bid[mid]
            if opts.weighting_least_sq == "var_spread":
                w_ls = 1.0 / spread**2
            else:
                w_ls = np.ones(mid.sum())
            q_j = float(w_ls.sum())
            w_mid = w_ls / mid.sum()

            if opts.bid_ask_penalty_scope == "one_sided":
                ask_set = has_ask & ~has_bid
                bid_set = has_bid & ~has_ask & in_range
            else:
                ask_set = has_ask
                bid_set = has_bid & in_range

            def side_weights(sel, vega_col, scheme, df=df, q_j=q_j):
                n = int(sel.sum())
                if n == 0:
                    return np.zeros(0)
                if scheme == "vega":
                    vega = np.nan_to_num(df[vega_col].to_numpy(dtype=float)[sel])
                    raw = q_j * vega / max(vega.sum(), 1e-300)
                else:
                    raw = np.full(n, q_j / n)
                return raw / n

            expiries.append(
                _Expiry(
                    expiry=exp,
                    t_e=t_e,
                    fwd=fwd,
                    sigma_star=float(sigma_star),
                    z_mid=z[mid],
                    v_mid=0.5 * (v_bid[mid] + v_ask[mid]),
                    w_mid=w_mid,
                    z_ask=z[ask_set],
                    v_ask=v_ask[ask_set],
                    w_ask=side_weights(ask_set, "vega_ask", opts.weighting_above_ask),
                    z_bid=z[bid_set],
                    v_bid=v_bid[bid_set],
                    w_bid=side_weights(bid_set, "vega_bid", opts.weighting_below_bid),
                    n_crossed=int(crossed.sum()),
                )
            )
        expiries.sort(key=lambda e: e.t_e)
        return expiries, dropped

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
        fitter_options: CviVolFitterOptions | None = None,
    ) -> CviFitResult:
        opts = fitter_options or CviVolFitterOptions()
        basis = _Basis(node_locs)
        exps, dropped = self._prepare(chain, basis, opts)
        if not exps:
            raise ValueError("No expiry has enough quotes to be fitted")

        m, n_nodes = basis.m, len(node_locs)
        # variable layout: [alpha_0 .. alpha_{J-1} | per expiry: s_ask, s_bid, u_tv]
        offsets_alpha = [j * m for j in range(len(exps))]
        cursor = len(exps) * m
        slack_offsets = []
        for e in exps:
            o_ask = cursor
            o_bid = o_ask + len(e.z_ask)
            o_tv = o_bid + len(e.z_bid)
            cursor = o_tv + (n_nodes - 1)
            slack_offsets.append((o_ask, o_bid, o_tv))
        qp = _QP(cursor)
        B1_z0, B1_zn = basis.rows(basis.z0, 1), basis.rows(basis.zn, 1)
        B2_z0, B2_zn = basis.rows(basis.z0, 2), basis.rows(basis.zn, 2)
        B2_knots = basis.rows(basis.locs, 2)
        D_tv = B2_knots[1:] - B2_knots[:-1]

        for j, e in enumerate(exps):
            oa = offsets_alpha[j]
            o_ask, o_bid, o_tv = slack_offsets[j]
            # --- least squares on mid variances
            qp.add_quadratic(qp.block(basis.rows(e.z_mid, 0), oa), e.v_mid, e.w_mid)
            # --- above ask: s >= B a - v_ask, s >= 0, penalty w s^2
            if len(e.z_ask):
                n_a = len(e.z_ask)
                I = np.eye(n_a)
                qp.add_quadratic(qp.block(I, o_ask), np.zeros(n_a), e.w_ask)
                qp.le(
                    qp.block(basis.rows(e.z_ask, 0), oa) + qp.block(-I, o_ask), e.v_ask
                )
                qp.le(qp.block(-I, o_ask), np.zeros(n_a))
            # --- below bid: s >= v_bid - B a, s >= 0
            if len(e.z_bid):
                n_b = len(e.z_bid)
                I = np.eye(n_b)
                qp.add_quadratic(qp.block(I, o_bid), np.zeros(n_b), e.w_bid)
                qp.le(
                    qp.block(-basis.rows(e.z_bid, 0), oa) + qp.block(-I, o_bid),
                    -e.v_bid,
                )
                qp.le(qp.block(-I, o_bid), np.zeros(n_b))
            # --- strike regularization: lambda * sum |c_i - c_{i+1}|
            lam = opts.strike_regularization_factor
            n_tv = n_nodes - 1
            I = np.eye(n_tv)
            if lam > 0:
                qp.q[o_tv : o_tv + n_tv] += lam
                qp.le(
                    qp.block(D_tv / e.v_star, oa) + qp.block(-I, o_tv), np.zeros(n_tv)
                )
                qp.le(
                    qp.block(-D_tv / e.v_star, oa) + qp.block(-I, o_tv), np.zeros(n_tv)
                )
            else:
                qp.eq(qp.block(I, o_tv), np.zeros(n_tv))

            # --- boundary: zero convexity at edge knots (linear wings)
            qp.eq(qp.block(B2_z0, oa), 0.0)
            qp.eq(qp.block(B2_zn, oa), 0.0)
            # --- increasing variance in the wings (eq. 1)
            qp.le(qp.block(B1_z0, oa), 0.0)
            qp.le(qp.block(-B1_zn, oa), 0.0)
            # --- Lee's tail slope bounds (eq. 2): sqrt(T)/(2 sigma*) |dv/dz| < 1
            lee = (1.0 - opts.lee_margin) * 2.0 * e.sigma_star / np.sqrt(e.t_e)
            qp.le(qp.block(-B1_z0, oa), lee)
            qp.le(qp.block(B1_zn, oa), lee)

        # --- positivity of the variance on the first expiry
        z_pos = np.linspace(basis.z0, basis.zn, opts.n_positivity_points)
        qp.le(
            qp.block(-basis.rows(z_pos, 0), offsets_alpha[0]),
            -1e-8 * np.ones(len(z_pos)),
        )

        # --- calendar spread: T_j v_j(k) <= T_{j+1} v_{j+1}(k) at fixed k = log(K/F)
        z_cal = np.linspace(basis.z0, basis.zn, opts.n_calendar_points)
        for j in range(len(exps) - 1):
            e1, e2 = exps[j], exps[j + 1]
            o1, o2 = offsets_alpha[j], offsets_alpha[j + 1]
            k = np.concatenate([z_cal * e1.z_to_k, z_cal * e2.z_to_k])
            rows1 = e1.t_e * basis.rows(k / e1.z_to_k, 0)
            rows2 = e2.t_e * basis.rows(k / e2.z_to_k, 0)
            qp.le(qp.block(rows1, o1) + qp.block(-rows2, o2), np.zeros(len(k)))
            # tails (eq. 4): slope of total variance in k ordered in T
            a1, a2 = np.sqrt(e1.t_e) / e1.sigma_star, np.sqrt(e2.t_e) / e2.sigma_star
            qp.le(qp.block(-a1 * B1_z0, o1) + qp.block(a2 * B1_z0, o2), 0.0)
            qp.le(qp.block(a1 * B1_zn, o1) + qp.block(-a2 * B1_zn, o2), 0.0)

        # --- sequential solves with linearized butterfly constraints
        statuses, times, violations = [], [], []
        bf_rows: list[tuple[sparse.csr_matrix, FloatArr]] = []
        # interior grid plus interior knots, where the piecewise-linear convexity kinks
        # and a denser set in the two edge segments, next to the edge-knot skew bounds
        z_bf = np.unique(
            np.concatenate(
                [
                    np.linspace(basis.z0, basis.zn, opts.n_butterfly_points + 2)[1:-1],
                    basis.locs[1:-1],
                    np.linspace(basis.z0, basis.locs[1], 17)[1:-1],
                    np.linspace(basis.locs[-2], basis.zn, 17)[1:-1],
                ]
            )
        )
        z_chk = np.linspace(basis.z0, basis.zn, opts.n_check_points)
        rows_chk = [basis.rows(z_chk, d) for d in range(3)]
        rows_bf = [basis.rows(z_bf, d) for d in range(3)]
        x = np.zeros(qp.n_var)
        obj = np.nan
        for it in range(opts.n_iter_max):
            status, x, obj, elapsed = qp.solve(bf_rows, opts.verbose)
            statuses.append(status)
            times.append(elapsed)
            n_viol = 0
            for j, e in enumerate(exps):
                a = x[offsets_alpha[j] : offsets_alpha[j] + m]
                v, dv, d2v = (r @ a for r in rows_chk)
                s, c = dv / e.v_star, d2v / e.v_star
                gv = bf.g(np.maximum(v, 1e-12), s, z_chk * e.z_to_k, e.v_star, e.t_e)
                n_viol += int(np.sum(c - gv < -opts.butterfly_tolerance))
            violations.append(n_viol)
            if status != "Solved" or n_viol == 0 or it == opts.n_iter_max - 1:
                break
            bf_rows = self._butterfly_rows(
                qp, basis, exps, offsets_alpha, x, z_bf, rows_bf
            )

        self._slices = {}
        for j, e in enumerate(exps):
            a = x[offsets_alpha[j] : offsets_alpha[j] + m]
            spline = CviCubicBSplineParams(basis.knots.copy(), a.copy())
            self._slices[e.expiry] = CviSlice.from_spline_params(
                spline, e.fwd, e.t_e, e.v_star
            )
        return CviFitResult(
            slices=dict(self._slices),
            statuses=statuses,
            solve_times=times,
            butterfly_violations=violations,
            n_iterations=len(statuses),
            objective=obj,
            dropped_expiries=dropped,
            n_crossed_quotes=sum(e.n_crossed for e in exps),
        )

    @staticmethod
    def _butterfly_rows(qp, basis, exps, offsets_alpha, x, z_bf, rows_bf):
        """Linearized no-butterfly-arbitrage constraints (eqs. 5, 18, 19)."""
        out = []
        m = basis.m
        B0e = [basis.rows(basis.z0, 0), basis.rows(basis.zn, 0)]
        B1e = [basis.rows(basis.z0, 1), basis.rows(basis.zn, 1)]
        for j, e in enumerate(exps):
            oa = offsets_alpha[j]
            a = x[oa : oa + m]
            B0, B1, B2 = rows_bf
            v_ref = np.maximum(B0 @ a, 1e-8)
            s_ref = (B1 @ a) / e.v_star
            k = z_bf * e.z_to_k
            b0, b1, b2 = bf.linearized_interior(v_ref, s_ref, k, e.v_star, e.t_e)
            # c >= b0 + b1 (s - s_ref) + b2 (v - v_ref)
            A = -B2 / e.v_star + (b1[:, None] * B1) / e.v_star + b2[:, None] * B0
            out.append((qp.block(A, oa), -b0 + b1 * s_ref + b2 * v_ref))
            # edge knots, where c = 0
            for side, z_e, B0_, B1_ in (
                ("left", basis.z0, B0e[0], B1e[0]),
                ("right", basis.zn, B0e[1], B1e[1]),
            ):
                v_e = max(float((B0_ @ a)[0]), 1e-8)
                k_e = z_e * e.z_to_k
                if bf.d1d2(v_e, k_e, e.t_e) > 1.0:
                    bound, slope = bf.linearized_edge(v_e, k_e, e.v_star, e.t_e, side)
                    if side == "left":  # s >= bound + slope (v - v_e)
                        A_e = -B1_ / e.v_star + slope * B0_
                        b_e = -bound + slope * v_e
                    else:  # s <= bound + slope (v - v_e)
                        A_e = B1_ / e.v_star - slope * B0_
                        b_e = bound - slope * v_e
                else:  # fall back to eq. 5 with c fixed at 0
                    s_e = float((B1_ @ a)[0]) / e.v_star
                    c0, c1, c2 = bf.linearized_interior(
                        np.array([v_e]),
                        np.array([s_e]),
                        np.array([k_e]),
                        e.v_star,
                        e.t_e,
                    )
                    A_e = (c1[0] * B1_) / e.v_star + c2[0] * B0_
                    b_e = -c0[0] + c1[0] * s_e + c2[0] * v_e
                out.append((qp.block(A_e, oa), np.atleast_1d(b_e)))
        return out
