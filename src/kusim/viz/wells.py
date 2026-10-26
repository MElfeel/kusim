"""Well production and pressure profiles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..results import Results
from .fields import pick_snapshot


def _selected(results: Results, wells):
    frame = results.wells
    if frame.empty:
        raise ValueError("this run has no wells, so there are no well profiles to plot")
    names = list(dict.fromkeys(frame["well"]))
    if wells is not None:
        unknown = [w for w in wells if w not in names]
        if unknown:
            raise ValueError(f"unknown well(s) {unknown}; available: {names}")
        names = list(wells)
    return names


def plot_wells(results: Results, wells=None):
    """Four-panel summary: rates, bottom-hole pressure, water cut and cumulative volumes.

    Producers are drawn with solid lines, injectors with dashed ones.  ``wells`` can
    be a list of well names such as ``["P1"]``.
    """
    names = _selected(results, wells)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    (ax_rate, ax_bhp), (ax_cut, ax_cum) = axes
    for name in names:
        w = results.well(name)
        if w.kind.iloc[0] == "producer":
            ax_rate.plot(w.time, w.oil_rate, label=f"{name} oil")
            ax_rate.plot(w.time, w.water_rate, label=f"{name} water")
            ax_cut.plot(w.time, w.water_cut, label=name)
            ax_cum.plot(w.time, w.cum_oil, label=f"{name} oil")
            ax_cum.plot(w.time, w.cum_water, label=f"{name} water")
        else:
            ax_rate.plot(w.time, w.water_rate, "--", label=f"{name} injection")
            ax_cum.plot(w.time, w.cum_water, "--", label=f"{name} injected")
        ax_bhp.plot(w.time, w.bhp, "--" if w.kind.iloc[0] == "injector" else "-", label=name)
    ax_rate.set(ylabel="rate (STB/day)", title="Surface rates")
    ax_bhp.set(ylabel="pressure (psi)", title="Bottom-hole pressure")
    ax_cut.set(ylabel="water cut (fraction)", title="Water cut", ylim=(0, 1), xlabel="time (days)")
    ax_cum.set(ylabel="volume (STB)", title="Cumulative volumes", xlabel="time (days)")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pressure_history(results: Results, cells=None, ax=None):
    """Oil pressure against time in chosen blocks (thesis block numbers; default: the well blocks)."""
    g = results.grid
    if cells is None:
        cells = sorted(set(results.wells["block"])) if not results.wells.empty else [1]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for block in cells:
        ax.plot(results.times, results.pressure[:, g.cell_index(block)], label=f"block {block}")
    ax.set(xlabel="time (days)", ylabel="oil pressure (psi)", title="Block pressures")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig if fig is not None else ax


# ----------------------------------------------------------------------------
# Wells completed in several layers
# ----------------------------------------------------------------------------
def _perforation_table(results: Results, well: str):
    perf = results.perforations_of(well) if not results.perforations.empty else results.perforations
    if perf.empty:
        known = list(dict.fromkeys(results.wells["well"])) if not results.wells.empty else []
        raise ValueError(f"no perforation data for well {well!r}; wells in this run: {known}")
    return perf


def _layer_color(k):
    return plt.get_cmap("tab10")((k - 1) % 10)


def plot_inflow_profile(results: Results, well: str, snapshot=-1, time=None, ax=None):
    """Bar chart of how a well draws from (or injects into) each layer it is completed in.

    One pair of bars per perforation (oil and water for a producer, water for an injector),
    with the layer at the top of the chart for layer 1.  The percentage is the perforation's
    share of the well's total liquid (or injected water).
    """
    perf = _perforation_table(results, well)
    t_wanted = results.times[pick_snapshot(results, snapshot, time)]
    steps = perf["time"].unique()
    t = steps[int(np.argmin(np.abs(steps - t_wanted)))]            # nearest time step with rates
    rows = perf[perf["time"] == t].sort_values("k").reset_index(drop=True)
    producer = rows["kind"].iloc[0] == "producer"
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 1.2 + 0.9 * len(rows)))
    y = np.arange(len(rows))
    if producer:
        ax.barh(y - 0.2, rows["oil_rate"], height=0.38, color="tab:brown", label="oil")
        ax.barh(y + 0.2, rows["water_rate"], height=0.38, color="tab:blue", label="water")
        total = (rows["oil_rate"] + rows["water_rate"]).sum()
        share = (rows["oil_rate"] + rows["water_rate"]) / total if total else rows["oil_rate"] * 0
        ax.legend(loc="upper right")
    else:
        ax.barh(y, rows["water_rate"], height=0.6, color="tab:blue", label="water injected")
        total = rows["water_rate"].sum()
        share = rows["water_rate"] / total if total else rows["water_rate"] * 0
    biggest = max(rows[["oil_rate", "water_rate"]].abs().max().max(), 1e-12)
    for n, (_, r) in enumerate(rows.iterrows()):
        rate = max(r["oil_rate"], r["water_rate"], key=abs) if producer else r["water_rate"]
        ax.text(max(rate, 0) + 0.02 * biggest, n, f"{100 * share[n]:.0f} %", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"layer {r.k}\n(block {r.block})" for r in rows.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set(xlabel="rate (STB/day)", title=f"{well}: rate by layer, t = {t:g} d")
    ax.set_xlim(right=biggest * 1.3)
    ax.grid(axis="x", alpha=0.3)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax


def plot_perforation_rates(results: Results, well: str):
    """Stacked rates by layer through time: which layers a multi-layer well draws from, and when.

    A producer gets two panels (oil and water), an injector one.  Layers keep the same colour in
    every panel.  The thick line is the well total.
    """
    perf = _perforation_table(results, well)
    producer = perf["kind"].iloc[0] == "producer"
    columns = [("oil_rate", "Oil rate (STB/day)"), ("water_rate", "Water rate (STB/day)")] if producer \
        else [("water_rate", "Water injection rate (STB/day)")]
    fig, axes = plt.subplots(1, len(columns), figsize=(5.6 * len(columns), 4), squeeze=False)
    layers = perf.drop_duplicates("block").sort_values("k")[["block", "k"]]
    for ax, (column, label) in zip(axes[0], columns):
        table = perf.pivot(index="time", columns="block", values=column)[list(layers["block"])]
        ax.stackplot(table.index, table.T.values,
                     labels=[f"layer {k} (block {b})" for b, k in zip(layers["block"], layers["k"])],
                     colors=[_layer_color(k) for k in layers["k"]], alpha=0.85)
        total = table.sum(axis=1)
        ax.plot(table.index, total, color="k", lw=1.2)
        ax.set(xlabel="time (days)", ylabel=label, title=f"{well}: {label.split(' (')[0].lower()} by layer")
        # A start-up transient (the well opens against the initial pressure) can be many times
        # larger than the rest of the history; cut the axis so the rest stays readable.
        typical = float(np.percentile(total.abs(), 95))
        if typical > 0 and total.abs().max() > 3 * typical:
            ax.set_ylim(0, 1.25 * typical)
            ax.annotate("start-up transient\noff scale", (table.index[0], 1.25 * typical),
                        xytext=(8, -26), textcoords="offset points", fontsize=8, color="0.3")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig
