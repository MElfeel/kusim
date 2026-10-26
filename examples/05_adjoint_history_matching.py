"""History matching with adjoint gradients on the twin model; writes a PNG next to this file.

    py examples/05_adjoint_history_matching.py

The same truth, data and prior as the Ensemble Kalman Filter of ``04_enkf_twin_experiment.py``, matched by L-BFGS-B with
the gradient of the objective from the adjoint (``kusim.adjoint``): one gradient costs about three simulations, however
many of the 72 permeabilities are unknown.  The prior (the mean and covariance that generated the twin) turns the problem
into a maximum a posteriori estimate.  The adjoint gradient is first checked against finite differences.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adjoint_workflows import history_match
from enkf_twin import run_twin_experiment

HERE = Path(__file__).resolve().parent
GREY, BLUE, ORANGE, BLACK = "#9a9a9a", "#0072B2", "#D55E00", "#222222"


def plot(out, path):
    twin, grid, obs = out["twin"], out["twin"]["grid"], out["twin"]["observations"].table
    truth, start, end = np.log(twin["truth_model"].kx), out["mean"], out["x"]
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], wspace=0.38, hspace=0.42)
    lo, hi = min(truth.min(), start.min(), end.min()), max(truth.max(), start.max(), end.max())
    layer = 1
    for col, (title, values) in enumerate([("truth", truth), ("prior mean (start)", start), ("adjoint MAP", end)]):
        ax = fig.add_subplot(gs[0, col])
        image = ax.imshow(grid.to_3d(values)[layer], origin="lower", cmap="viridis", vmin=lo, vmax=hi, extent=(0, grid.nx * 400, 0, grid.ny * 400))
        ax.set_title(f"ln k, layer {layer + 1}: {title}", loc="left", fontsize=10)
        ax.set_xlabel("x (ft)")
        ax.grid(False)
        for name, (i, j) in {"P1": (1, 1), "I1": (6, 4)}.items():
            ax.plot((i - 0.5) * 400, (j - 0.5) * 400, "o", color="white", mec=BLACK, ms=7)
            ax.annotate(name, ((i - 0.5) * 400, (j - 0.5) * 400), xytext=(5, 5), textcoords="offset points", color="white", fontsize=9)
    fig.colorbar(image, ax=fig.axes[:3], shrink=0.8, pad=0.04, label="ln k (md)")

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(np.arange(1, len(out["history"]) + 1), out["history"], color=BLUE, lw=1.8)
    ax.axhline(len(obs), color=GREY, ls=":", label="noise level (one per datum)")
    ax.set_xlabel("function and gradient evaluations")
    ax.set_ylabel("data misfit (sum of residual / sigma squared)")
    ax.legend(fontsize=8)
    for col, (vector, label) in enumerate([("WBHP:P1", "producer BHP (psia)"), ("WWCT:P1", "producer water cut")], start=1):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(out["start_table"].TIME, out["start_table"][vector], color=GREY, lw=1.6, label="start (prior mean)")
        ax.plot(out["end_table"].TIME, out["end_table"][vector], color=BLUE, lw=1.8, label="adjoint MAP")
        ax.plot(twin["truth"].table.TIME, twin["truth"].table[vector], color=BLACK, ls="--", lw=1.5, label="truth")
        rows = obs[obs.vector == vector]
        ax.errorbar(rows.time, rows.value, yerr=rows.sigma, fmt="o", color=BLACK, ms=4, capsize=2, label="data (noisy)")
        ax.set_xlabel("time (days)")
        ax.set_ylabel(label)
        if col == 1:
            ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    out = history_match(seed=0, time_finite_differences=True)
    m, check = out["metrics"], out["check"]
    print("gradient check at the start (adjoint against finite differences through the simulator):")
    print(f"  {int(check.stable.sum())} of {len(check)} components have a stable finite-difference estimate; "
          f"largest relative difference {check.relative.max():.1e}")
    t_run, t_adj, t_fd = out["seconds_per_run"], out["seconds_per_gradient"], out["seconds_per_fd_gradient"]
    print(f"cost of one gradient for {out['problem'].n_params} parameters: adjoint {t_adj:.2f} s ({t_adj / t_run:.1f} simulations), "
          f"finite differences {t_fd:.1f} s ({t_fd / t_run:.0f} simulations)")
    print(f"L-BFGS-B: {out['iterations']} iterations, {out['evaluations']} evaluations, {out['seconds']:.0f} s ({out['message'].strip()})")
    print(f"data misfit (RMS of residual / sigma): start {m['start_data_rms']:.1f}, adjoint estimate {m['end_data_rms']:.2f}")
    print(f"ln k error against the truth:          start {m['start_logk_rmse']:.2f}, adjoint estimate {m['end_logk_rmse']:.2f}")
    print("layer means of ln k: truth", np.round(m["truth_layers"], 2), "| start", np.round(m["start_layers"], 2),
          "| estimate", np.round(m["end_layers"], 2))

    enkf = run_twin_experiment(ne=30, seed=0)["metrics"]
    adjoint_runs = out["evaluations"] * t_adj / t_run
    print(f"\nfor comparison, the EnKF of example 04 on the same twin (30 members, about 90 simulations in all): "
          f"ln k error of the ensemble mean {enkf['posterior_logk_rmse']:.2f}.")
    print(f"The adjoint optimisation used about {adjoint_runs:.0f} simulations. Note that it was given the exact prior mean and "
          f"covariance, while the EnKF had only a 30-member sample of them.")
    path = HERE / "adjoint_history_matching.png"
    plot(out, path)
    print("saved", path.name)


if __name__ == "__main__":
    main()
