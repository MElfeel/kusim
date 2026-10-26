"""ES-MDA twin experiment on the history-matching API; writes a PNG next to this file.

    py examples/07_esmda_twin_experiment.py

A truth model with a known permeability field gives noisy well data (the same truth, data and prior as the Ensemble Kalman
Filter of ``04_enkf_twin_experiment.py``).  An ensemble of 40 prior models is history matched with the Ensemble Smoother with
Multiple Data Assimilation: four passes, each running all models over the whole history, comparing with all the data and
improving the permeability a little (see ``esmda.py`` for the method, ``esmda_twin.py`` for the workflow).  The figure shows
the permeability of the middle layer (truth, prior mean, posterior mean) and the producer pressure, injector pressure and
producer water cut of the prior ensemble, of the updated ensemble, and the data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from esmda_twin import run_esmda_experiment

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
            ax.plot(record.table.TIME, record.table[vector], color=BLUE, lw=0.8, alpha=0.7, label="updated members" if k == 0 else None)
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
    out = run_esmda_experiment(ne=40, seed=0, n_passes=4)
    m = out["metrics"]
    print(f"members: {m['members']}, passes: {len(m['alphas'])} (alpha {m['alphas']}), data dates: {len(out['dates'])}")
    print("data misfit (RMS of residual / sigma, mean over members) before and after each pass:",
          " -> ".join(f"{x:.1f}" for x in m["rms_by_pass"]))
    print(f"ln k error of the ensemble mean against the truth: prior {m['prior_logk_rmse']:.2f}, updated {m['posterior_logk_rmse']:.2f}")
    print(f"ensemble spread of ln k (mean over blocks):        prior {m['prior_spread']:.2f}, updated {m['posterior_spread']:.2f}")
    print(f"members flagged by the diagnostics before and after each pass: {m['not_ok_by_pass']}; at the end {m['members'] - m['posterior_ok']}")
    print("history of a member:", out["posterior_models"][0].provenance[:2], "...", len(out["posterior_models"][0].provenance), "edits")
    path = HERE / "esmda_twin_experiment.png"
    plot(out, path)
    print("saved", path.name)


if __name__ == "__main__":
    main()
