"""Fit the CVI surface to the bundled S&P 500 snapshot (2023-04-14).

Run from the repository root:  python -m cvi.examples.fit_spx [--plot]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import numpy as np
import pandas as pd

from cvi.fitter.fitter import CviVolFitter
from cvi.fitter.fitter_options import CviVolFitterOptions

DATA = Path(__file__).parent / "data" / "2023-04-14_vol.csv"
NODES = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]


def main(plot: bool = False) -> None:
    chain = pd.read_csv(DATA, parse_dates=["expiry"])
    res = CviVolFitter().fit(chain, NODES, CviVolFitterOptions())
    print(f"expiries fitted: {len(res.slices)}, dropped: {len(res.dropped_expiries)}")
    print(f"crossed quotes ignored: {res.n_crossed_quotes}")
    print(f"solver statuses: {res.statuses}")
    print(f"butterfly violations after each solve: {res.butterfly_violations}")
    print(f"total Clarabel time: {sum(res.solve_times):.3f}s")

    if plot:
        import matplotlib.pyplot as plt

        exps = sorted(res.slices)[:: max(1, len(res.slices) // 6)]
        fig, axes = plt.subplots(len(exps), 1, figsize=(8, 2.6 * len(exps)))
        for ax, exp in zip(np.atleast_1d(axes), exps):
            slc = res.slices[exp]
            df = chain[chain["expiry"] == exp]
            z = slc.k_to_z(df["strike"].to_numpy())
            sel = (z >= NODES[0]) & (z <= NODES[-1])
            zz = np.linspace(z[sel].min(), z[sel].max(), 400)
            ax.plot(zz, np.sqrt(slc.var_deriv1_deriv2_z(zz)[0]), "g-", label="CVI")
            ax.plot(z[sel], df["iv_bid"].to_numpy()[sel], "r_", label="bid")
            ax.plot(z[sel], df["iv_ask"].to_numpy()[sel], "b_", label="ask")
            ax.set_title(str(exp))
            ax.set_xlabel("z")
        np.atleast_1d(axes)[0].legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
