"""Production optimisation with adjoint gradients; writes a PNG next to this file.

    py examples/06_adjoint_production_optimisation.py

A fixed volume of injection water (an average of 9000 STB/day over 600 days) is distributed over four periods of 150 days to
maximise the profit at 600 days (oil 60, produced water -20, injected water -3 per STB).  The gradient of the profit with
respect to the four injection rates comes from the adjoint; SLSQP uses it with bounds on the rates and the volume as an
equality constraint.  A derivative-free search (Nelder-Mead) on the same problem is the baseline, and the result is checked
by running the simulator again.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adjoint_workflows import PERIODS, allocate_injection
from kusim.history import run_to

HERE = Path(__file__).resolve().parent
GREY, BLUE, BLACK = "#9a9a9a", "#0072B2", "#222222"


def plot(out, path):
    problem = out["problem"]
    days = list(np.arange(10.0, 601.0, 10.0))
    vectors = ["WOPR:P1", "WWCT:P1", "FOPT"]
    runs = {name: run_to(*(lambda m_s: (m_s[0], days, m_s[1]))(problem.build(x)), vectors=vectors).table
            for name, x in (("uniform start", out["x0"]), ("optimised", out["x"]))}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axes[0, 0]
    edges = PERIODS + [600.0]
    ax.stairs(out["x0"], edges, color=GREY, ls=":", lw=2, label="uniform start")
    ax.stairs(out["x"], edges, color=BLUE, lw=2, label="optimised (SLSQP, adjoint)")
    if "nelder_mead" in out:
        ax.stairs(out["nelder_mead"]["x"], edges, color=BLACK, ls="--", lw=1.4, label="Nelder-Mead")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("water injection rate (STB/day)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(np.arange(1, len(out["history"]) + 1), np.array(out["history"]) / 1e6, "o-", color=BLUE, lw=1.8, ms=4, label="SLSQP with adjoint gradient")
    if "nelder_mead" in out:
        ax.axhline(out["nelder_mead"]["profit"] / 1e6, color=BLACK, ls="--", label=f"Nelder-Mead ({out['nelder_mead']['evaluations']} evaluations)")
    ax.set_xlabel("function and gradient evaluations")
    ax.set_ylabel("profit (M$)")
    ax.legend(fontsize=8)
    for ax, vector, label in ((axes[1, 0], "WOPR:P1", "oil rate (STB/day)"), (axes[1, 1], "WWCT:P1", "producer water cut")):
        for name, table in runs.items():
            ax.plot(table.TIME, table[vector], color=GREY if name == "uniform start" else BLUE, ls=":" if name == "uniform start" else "-", lw=2, label=name)
        ax.set_xlabel("time (days)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    out = allocate_injection(compare_nelder_mead=True)
    check = out["check"]
    print(f"gradient check at the start: {int(check.stable.sum())} of {len(check)} components stable, largest relative difference {check.relative.max():.1e}")
    print(f"profit at 600 days: uniform {out['profit_start'] / 1e6:.2f} M$, optimised {out['profit_end'] / 1e6:.2f} M$ (by running the simulator again)")
    print(f"SLSQP with the adjoint: {out['iterations']} iterations, {out['evaluations']} evaluations, {out['seconds']:.0f} s ({out['message']})")
    nm = out["nelder_mead"]
    print(f"Nelder-Mead from the same start: {nm['profit'] / 1e6:.2f} M$ with {nm['evaluations']} evaluations")
    print("optimised injection rates (STB/day):", np.round(out["x"]), "| Nelder-Mead:", np.round(nm["x"]), "| total", round(out["x"].sum()))
    print(f"along the whole path of the optimiser: backflow well-steps {out['backflow_on_the_path']}, producer mode switches {out['mode_switches_on_the_path']}")
    gain = out["profit_end"] / out["profit_start"] - 1.0
    print(f"The gain is {100 * gain:.1f} %: inject early and taper, the rest of the allocation matters little here.")
    path = HERE / "adjoint_production_optimisation.png"
    plot(out, path)
    print("saved", path.name)


if __name__ == "__main__":
    main()
