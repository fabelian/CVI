"""Calibration time versus number of knots (cf. Table 1 of the CVI paper).

Run from the repository root:  python -m cvi.examples.benchmark [n_repeat]
Prints a Markdown table for the bundled S&P 500 snapshot.
"""

import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import numpy as np
import pandas as pd

from cvi.fitter.fitter import CviVolFitter
from cvi.fitter.fitter_options import CviVolFitterOptions

DATA = Path(__file__).parent / "data" / "2023-04-14_vol.csv"
Z_MAX = 6.0


def make_nodes(n: int, z_max: float = Z_MAX, a: float = 1.5) -> list[float]:
    """n knots on [-z_max, z_max], including 0, denser near the money (sinh map)."""
    n_left = (n - 1) // 2
    n_right = n - 1 - n_left

    def side(m):
        u = np.arange(1, m + 1) / m
        return z_max * np.sinh(a * u) / np.sinh(a)

    return sorted([*(-side(n_left)), 0.0, *side(n_right)])


def inside_bid_ask(res, chain) -> float:
    inside = total = 0
    for exp, slc in res.slices.items():
        df = chain[chain["expiry"] == exp]
        z = slc.k_to_z(df["strike"].to_numpy())
        bid, ask = df["iv_bid"].to_numpy(), df["iv_ask"].to_numpy()
        ok = (np.abs(z) <= Z_MAX) & np.isfinite(bid) & np.isfinite(ask) & (bid < ask)
        vol = np.sqrt(slc.var_deriv1_deriv2_z(z[ok])[0])
        inside += np.sum((vol >= bid[ok] - 1e-9) & (vol <= ask[ok] + 1e-9))
        total += ok.sum()
    return inside / total


def main(n_repeat: int = 3) -> None:
    chain = pd.read_csv(DATA, parse_dates=["expiry"])
    n_quotes = int(chain["iv_bid"].notna().sum() + chain["iv_ask"].notna().sum())
    rows = []
    for n in [5, 10, 15, 20, 30, 40, 80]:
        nodes = make_nodes(n)
        best = None
        for _ in range(n_repeat):
            start = time.perf_counter()
            res = CviVolFitter().fit(chain, nodes, CviVolFitterOptions())
            wall = time.perf_counter() - start
            solver = sum(res.solve_times)
            if best is None or solver < best[1]:
                best = (res, solver, wall)
        res, solver, wall = best
        rows.append(
            {
                "knots": n,
                "iters": res.n_iterations,
                "bfly viol.": "→".join(map(str, res.butterfly_violations)),
                "P nnz": res.p_nnz[0],
                "A nnz (1st / last)": f"{res.a_nnz[0]:,} / {res.a_nnz[-1]:,}",
                "constraints (1st / last)": f"{res.n_constraints[0]:,} / {res.n_constraints[-1]:,}",
                "solver 1st (s)": f"{res.solve_times[0]:.3f}",
                "solver total (s)": f"{solver:.3f}",
                "wall total (s)": f"{wall:.2f}",
                "in bid-ask": f"{100 * inside_bid_ask(res, chain):.1f}%",
                "status": "ok" if res.success else ",".join(res.statuses),
            }
        )
    table = pd.DataFrame(rows)
    print(
        f"S&P 500 2023-04-14: {len(res.slices)} expiries fitted, {n_quotes:,} vol bids or asks; "
        f"best of {n_repeat} runs; {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical CPUs\n"
    )
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
