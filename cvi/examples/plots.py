"""Figures in the style of the CVI paper for the bundled S&P 500 snapshot.

Run from the repository root:  python -m cvi.examples.plots [output_dir]
Writes PNG files (default: docs/images/).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from cvi.fitter.fitter import CviVolFitter
from cvi.fitter.fitter_options import CviVolFitterOptions

DATA = Path(__file__).parent / "data" / "2023-04-14_vol.csv"
NODES = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]

C_FIT, C_PDF, C_LO, C_HI = "#2a9d3c", "#4a7bd0", "#00b4c8", "#8e44ad"


def total_variance(slc, k):
    return slc._t_e * slc.var_deriv1_deriv2_z(np.asarray(k) / slc._z_denom)[0]


def density(slc, strikes):
    """Risk-neutral density d^2 C / d K^2 of undiscounted Black calls (forward = 1)."""
    fwd = slc._ref_fwd

    def call(K):
        w = total_variance(slc, np.log(K / fwd))
        sw = np.sqrt(w)
        d1 = (np.log(fwd / K) + 0.5 * w) / sw
        return fwd * norm.cdf(d1) - K * norm.cdf(d1 - sw)

    h = 1e-3 * strikes
    return (call(strikes + h) - 2 * call(strikes) + call(strikes - h)) / h**2


def pick(slices, days):
    exps = sorted(slices, key=lambda e: slices[e]._t_e)
    t = np.array([slices[e]._t_e for e in exps])
    return [exps[int(np.argmin(np.abs(t - d / 365)))] for d in days]


def plot_fit_pdf(res, chain, out):
    slices = res.slices
    order = sorted(slices, key=lambda e: slices[e]._t_e)
    chosen = pick(slices, [7, 35, 91, 245])
    fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 3.4 * len(chosen)))
    for ax, exp in zip(axes, chosen):
        slc = slices[exp]
        df = chain[chain["expiry"] == exp].sort_values("strike")
        K = df["strike"].to_numpy()
        z = slc.k_to_z(K)
        sel = (z >= NODES[0]) & (z <= NODES[-1])
        bid, ask = df["iv_bid"].to_numpy(), df["iv_ask"].to_numpy()
        two = sel & np.isfinite(bid) & np.isfinite(ask) & (bid < ask)
        one = sel & ~two
        ax.vlines(K[two], bid[two], ask[two], color="red", lw=1.2, label="bid-ask")
        ax.plot(K[one], bid[one], "_", color="grey", ms=5)
        ax.plot(K[one], ask[one], "_", color="grey", ms=5, label="one-sided quote")
        Kf = np.linspace(K[sel].min(), K[sel].max(), 600)
        kf = np.log(Kf / slc._ref_fwd)
        ax.plot(
            Kf,
            np.sqrt(total_variance(slc, kf) / slc._t_e),
            color=C_FIT,
            lw=1.8,
            label="CVI fit",
        )
        # calendar channel from the neighbouring expiries
        i = order.index(exp)
        if i > 0:
            lo = np.sqrt(total_variance(slices[order[i - 1]], kf) / slc._t_e)
            ax.plot(Kf, lo, color=C_LO, lw=0.9, ls="--", label="calendar bound (prev.)")
        if i < len(order) - 1:
            hi = np.sqrt(total_variance(slices[order[i + 1]], kf) / slc._t_e)
            ax.plot(Kf, hi, color=C_HI, lw=0.9, ls="--", label="calendar bound (next)")
        ax.set_ylabel("implied vol")
        top = np.nanpercentile(ask[sel], 97)
        ax.set_ylim(0.9 * np.nanmin(bid[two]), 1.15 * top)
        ax.set_title(
            f"SPX {exp:%Y-%m-%d %H:%M}  (T = {slc._t_e * 365:.0f} days)", fontsize=10
        )
        ax2 = ax.twinx()
        pdf = density(slc, Kf)
        ax2.fill_between(Kf, 0, pdf, color=C_PDF, alpha=0.18, lw=0)
        ax2.set_ylim(bottom=0)
        ax2.set_yticks([])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("strike")
    fig.suptitle(
        "CVI fit, bid-asks and risk-neutral density (shaded), S&P 500 2023-04-14",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_total_variance(res, out):
    slices = res.slices
    order = sorted(slices, key=lambda e: slices[e]._t_e)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (k0, k1) in zip(axes, [(-1.2, 0.5), (-0.15, 0.08)]):
        k = np.linspace(k0, k1, 500)
        for i, e in enumerate(order):
            ax.plot(
                k,
                total_variance(slices[e], k),
                color=cmap(i / (len(order) - 1)),
                lw=0.8,
            )
        ax.set_xlabel("log-moneyness  k = log(K/F)")
        ax.set_ylabel("total variance  v T")
        ax.grid(alpha=0.3)
    axes[0].set_title("all expiries (colour = maturity)", fontsize=10)
    axes[1].set_title("zoom near the money: curves never cross", fontsize=10)
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_shape(res, out):
    slices = res.slices
    chosen = pick(slices, [7, 35, 91, 245, 600])
    z = np.linspace(NODES[0], NODES[-1], 600)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for e in chosen:
        slc = slices[e]
        v, dv, d2v = slc.var_deriv1_deriv2_z(z)
        vs = slc.atm_anchor_var
        lab = f"{slc._t_e * 365:.0f}d"
        axes[0].plot(z, v / vs, label=lab)
        axes[1].plot(z, dv / vs)
        axes[2].plot(z, d2v / vs)
    for x in NODES:
        axes[2].axvline(x, color="grey", lw=0.4, alpha=0.5)
    axes[0].set_ylabel("f = v / v*")
    axes[1].set_ylabel("s = v' / v*")
    axes[2].set_ylabel("c = v'' / v*")
    axes[2].set_xlabel("z  (knots shown as vertical lines)")
    axes[0].legend(fontsize=8, ncol=5)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Shape function and its normalized derivatives (c is piecewise linear)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(out_dir: str = "docs/images") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chain = pd.read_csv(DATA, parse_dates=["expiry"])
    res = CviVolFitter().fit(chain, NODES, CviVolFitterOptions())
    plot_fit_pdf(res, chain, out / "spx_fit_pdf.png")
    plot_total_variance(res, out / "spx_total_variance.png")
    plot_shape(res, out / "spx_shape.png")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
