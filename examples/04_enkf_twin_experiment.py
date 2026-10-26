"""Ensemble Kalman Filter twin experiment on the kusim history-matching API; writes a PNG next to this file.

    py examples/04_enkf_twin_experiment.py

A truth model with a known permeability field gives noisy well data; an ensemble of 40 prior models is history
matched by assimilating the data every 50 days (see ``enkf_twin.py`` for the workflow and ``enkf_reference.py`` for the
analysis step).  The figure shows the permeability of the middle layer (truth, prior mean, posterior mean) and the
producer pressure and water cut of the prior ensemble, of the updated ensemble re-run from time zero, and the data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enkf_twin import run_twin_experiment

HERE = Path(__file__).resolve().parent
GREY, BLUE, BLACK = "#9a9a9a", "#0072B2", "#222222"


def plot(out, path):
    grid, params = out["grid"], out["params"]
    truth_logk = np.log(out["truth_model"].kx)
    prior_mean = params.matrix(out["prior_models"]).mean(axis=1)
    post_mean = params.matrix(out["posterior_models"]).mean(axis=1)
    obs = out["observations"].table

    fig = plt.figure(figsize=(12, 7.2))
    grid_spec = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], wspace=0.38, hspace=0.42)
    layer = 1                                                          # the middle layer, the highest permeability
    lo = min(truth_logk.min(), prior_mean.min(), post_mean.min())
    hi = max(truth_logk.max(), prior_mean.max(), post_mean.max())
    for col, (title, values) in enumerate([("truth", truth_logk), ("prior mean", prior_mean), ("posterior mean", post_mean)]):
        ax = fig.add_subplot(grid_spec[0, col])
        image = ax.imshow(grid.to_3d(values)[layer], origin="lower", cmap="viridis", vmin=lo, vmax=hi,
                          extent=(0, grid.nx * 400, 0, grid.ny * 400))
        ax.set_title(f"ln k, layer {layer + 1}: {title}", loc="left", fontsize=10)
        ax.set_xlabel("x (ft)")
        ax.grid(False)
        for name, (i, j) in {"P1": (1, 1), "I1": (6, 4)}.items():
            ax.plot((i - 0.5) * 400, (j - 0.5) * 400, "o", color="white", mec=BLACK, ms=7)
            ax.annotate(name, ((i - 0.5) * 400, (j - 0.5) * 400), xytext=(5, 5), textcoords="offset points", color="white", fontsize=9)
    fig.colorbar(image, ax=fig.axes[:3], shrink=0.8, pad=0.02, label="ln k (md)")

    for col, (vector, label) in enumerate([("WBHP:P1", "producer BHP (psia)"), ("WBHP:I1", "injector BHP (psia)"), ("WWCT:P1", "producer water cut")]):
        ax = fig.add_subplot(grid_spec[1, col])
        for k, record in enumerate(out["prior_records"]):
            ax.plot(record.table.TIME, record.table[vector], color=GREY, lw=0.8, alpha=0.7, label="prior members" if k == 0 else None)
        for k, record in enumerate(out["posterior_records"]):
            ax.plot(record.table.TIME, record.table[vector], color=BLUE, lw=0.8, alpha=0.7, label="updated members, re-run from day 0" if k == 0 else None)
        ax.plot(out["truth"].table.TIME, out["truth"].table[vector], color=BLACK, ls="--", lw=1.8, label="truth")
        rows = obs[obs.vector == vector]
        ax.errorbar(rows.time, rows.value, yerr=rows.sigma, fmt="o", color=BLACK, ms=4, capsize=2, label="data (noisy)")
        ax.set_xlabel("time (days)")
        ax.set_ylabel(label)
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    out = run_twin_experiment(ne=40, seed=0)
    m = out["metrics"]
    print(f"members: {m['members']}, assimilation dates: {len(out['dates'])}")
    print(f"data misfit (RMS of residual / sigma, mean over members): prior {m['prior_member_rms']:.1f}, updated and re-run {m['posterior_member_rms']:.1f}")
    print(f"ln k error of the ensemble mean against the truth:        prior {m['prior_logk_rmse']:.2f}, updated {m['posterior_logk_rmse']:.2f}")
    print(f"ensemble spread of ln k (mean over blocks):               prior {m['prior_spread']:.2f}, updated {m['posterior_spread']:.2f}")
    print(f"updated members without a diagnostics warning: {m['posterior_ok']} of {m['members']}; "
          f"saturations put back inside their limits during the updates: {m['saturations_clipped']}")
    print("history of a member:", out["posterior_models"][0].provenance[:2], "...", len(out["posterior_models"][0].provenance), "edits")
    path = HERE / "enkf_twin_experiment.png"
    plot(out, path)
    print("saved", path.name)


if __name__ == "__main__":
    main()
