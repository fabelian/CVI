"""No-butterfly-arbitrage conditions of CVI (Deschâtres, SSRN 4831218, Appendix B).

Notation (per expiry):
    z = k / (sigma_star * sqrt(T)),  k = log(K / F)
    v = sigma^2,  s = (1 / v_star) dv/dz,  c = (1 / v_star) d^2v/dz^2,  v_star = sigma_star^2

The risk-neutral density is non-negative iff c >= g(v, s; k), which is the
Gatheral condition on total variance w(k) = T v rewritten with
w' = sqrt(T v_star) s and w'' = c:

    (1 - k w' / (2 w))^2 - w'^2 / 4 (1 / w + 1 / 4) + w'' / 2 >= 0.
"""

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


def g(v: FloatArr, s: FloatArr, k: FloatArr, v_star: float, t_e: float) -> FloatArr:
    """Lower bound on the normalized convexity c (paper eq. 13)."""
    a = np.sqrt(t_e * v_star)
    w = t_e * v
    u = 1.0 - k * a * s / (2.0 * w)
    return 0.5 * a**2 * s**2 * (1.0 / w + 0.25) - 2.0 * u**2


def dg_ds(v: FloatArr, s: FloatArr, k: FloatArr, v_star: float, t_e: float) -> FloatArr:
    """Partial derivative of g with respect to s (beta_1 in paper eq. 5)."""
    a = np.sqrt(t_e * v_star)
    w = t_e * v
    u = 1.0 - k * a * s / (2.0 * w)
    return a**2 * s * (1.0 / w + 0.25) + 2.0 * u * k * a / w


def dg_dv(v: FloatArr, s: FloatArr, k: FloatArr, v_star: float, t_e: float) -> FloatArr:
    """Partial derivative of g with respect to v (beta_2 in paper eq. 5)."""
    a = np.sqrt(t_e * v_star)
    w = t_e * v
    u = 1.0 - k * a * s / (2.0 * w)
    dg_dw = -0.5 * a**2 * s**2 / w**2 - 2.0 * u * k * a * s / w**2
    return t_e * dg_dw


def linearized_interior(
    v_ref: FloatArr, s_ref: FloatArr, k: FloatArr, v_star: float, t_e: float
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Coefficients (beta_0, beta_1, beta_2) of c >= b0 + b1 (s - s_ref) + b2 (v - v_ref)."""
    return (
        g(v_ref, s_ref, k, v_star, t_e),
        dg_ds(v_ref, s_ref, k, v_star, t_e),
        dg_dv(v_ref, s_ref, k, v_star, t_e),
    )


def d1d2(v: float, k: float, t_e: float) -> float:
    return k * k / (v * t_e) - v * t_e / 4.0


def edge_skew_bound(v: float, k: float, v_star: float, t_e: float, side: str) -> float:
    """Skew bound at an edge knot where c = 0 (paper eqs. 18 and 19).

    side='left':  s >= s_min(v)   (z_0, both roots negative, take the higher root)
    side='right': s <= s_max(v)   (z_{n-1}, both roots positive, take the lower root)

    Valid when d1 * d2 > 1 at the edge knot.
    """
    sqrt_vt = np.sqrt(v * t_e)
    root = np.sqrt(1.0 + v * t_e / 4.0)
    denom = np.sqrt(v_star) * (k * k - v * v * t_e * t_e / 4.0 - v * t_e)
    sign = 1.0 if side == "left" else -1.0
    return float(2.0 * v * np.sqrt(t_e) * (k + sign * sqrt_vt * root) / denom)


def linearized_edge(
    v_ref: float, k: float, v_star: float, t_e: float, side: str
) -> tuple[float, float]:
    """(bound(v_ref), d bound / dv at v_ref) for the linearized edge constraint."""
    h = 1e-6 * max(v_ref, 1e-8)
    bound = edge_skew_bound(v_ref, k, v_star, t_e, side)
    slope = (
        edge_skew_bound(v_ref + h, k, v_star, t_e, side)
        - edge_skew_bound(v_ref - h, k, v_star, t_e, side)
    ) / (2.0 * h)
    return bound, slope
