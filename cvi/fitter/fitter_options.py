from typing import Literal

from pydantic import BaseModel, Field


class CviVolFitterOptions(BaseModel):
    """Options of the CVI calibration (Deschâtres, SSRN 4831218, Section 3)."""

    # Least-squares weights 1 / (v_ask - v_bid)^2 (chi-squared scaling) or uniform.
    weighting_least_sq: Literal["var_spread"] | None = Field(default="var_spread")
    # Above-ask / below-bid weights proportional to vega (normalized by q_j).
    weighting_above_ask: Literal["vega"] | None = Field(default="vega")
    weighting_below_bid: Literal["vega"] | None = Field(default="vega")
    # 'one_sided': penalize only options quoted on one side (paper's recommended
    # compromise, Section 3.2 tip); 'all': every option with an ask (bid).
    bid_ask_penalty_scope: Literal["one_sided", "all"] = Field(default="one_sided")
    # lambda of the strike regularization penalty (total variation of c).
    strike_regularization_factor: float = Field(default=0.05, ge=0.0)

    # Discretization of the constraints (number of z points per expiry).
    n_positivity_points: int = Field(default=20, ge=2)
    n_calendar_points: int = Field(default=40, ge=2)
    n_butterfly_points: int = Field(default=40, ge=1)

    # Lee's tail slope bound is enforced as |T dv/dk| <= 2 (1 - lee_margin).
    lee_margin: float = Field(default=1e-3, ge=0.0, lt=1.0)
    # Tolerance used to decide whether the fitted surface has butterfly arbitrage.
    butterfly_tolerance: float = Field(default=1e-6, ge=0.0)
    # Number of z points used to check for butterfly arbitrage after each solve.
    n_check_points: int = Field(default=400, ge=10)
    # Keep the linearized butterfly constraints of all previous solves (cutting
    # planes) instead of replacing them; avoids cycling of the linearization.
    accumulate_butterfly_cuts: bool = Field(default=True)
    # Total number of QP solves (first without butterfly constraints).
    n_iter_max: int = Field(default=6, ge=1)

    # Data filters.
    min_t_e: float = Field(default=1.0 / 365.0, ge=0.0)
    verbose: bool = Field(default=False)
