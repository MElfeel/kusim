"""Static pictures of the reservoir: layer maps, cross-sections and a 3-D block view.

All functions take a ``Results`` object and return the matplotlib figure (or axes
when you pass your own ``ax``), so you can keep customising them.  Pick the
field with ``field="sw"`` (water saturation), ``"so"`` or ``"pressure"``, and the
moment with ``snapshot`` (index into ``results.times``) or ``time`` (days).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from ..results import Results
from .fields import FIELDS, color_limits, cutaway_keep, field_data, pick_snapshot, well_completions

_MARKERS = {"producer": ("v", "tab:red"), "injector": ("^", "tab:blue")}


def _mark_wells(ax, results, layer, x_mid, y_mid, label_wells=True):
    """Mark every well that is completed in ``layer``.  ``x_mid``/``y_mid`` are the block centres."""
    for w in well_completions(results):
        here = [c for c in w["cells"] if c[2] == layer]
        if not here:
            continue
        marker, color = _MARKERS[w["kind"]]
        for i, j, _ in here:
            ax.plot(x_mid[i - 1], y_mid[j - 1], marker, color=color, mec="white", ms=9, zorder=5)
        if label_wells:
            i, j, _ = here[0]
            text = w["name"] if len(w["cells"]) == 1 else f"{w['name']} ({len(w['cells'])} layers)"
            ax.annotate(text, (x_mid[i - 1], y_mid[j - 1]), xytext=(5, 5), textcoords="offset points",
                        fontsize=8, weight="bold", zorder=6)


def plot_layer(results: Results, field="sw", layer=1, snapshot=-1, time=None, ax=None,
               vmin=None, vmax=None, wells=True, cmap=None, colorbar=True, title=None):
    """Map view of one layer (layer 1 is the top).  Blocks are drawn to scale.

    Producers are red triangles pointing down, injectors blue triangles pointing up.
    A well completed in several layers is marked in each of them.
    """
    g = results.grid
    if not 1 <= layer <= g.nz:
        raise ValueError(f"layer must be between 1 and {g.nz}")
    s = pick_snapshot(results, snapshot, time)
    lo, hi = color_limits(results, field, vmin, vmax)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    values = g.to_3d(field_data(results, field)[s])[layer - 1]
    mesh = ax.pcolormesh(g.x_edges, g.y_edges, values, cmap=cmap or FIELDS[field].cmap,
                         vmin=lo, vmax=hi, edgecolors="0.7", linewidth=0.4)
    ax.set_aspect("equal")
    ax.set(xlabel="x (ft)", ylabel="y (ft)",
           title=title or f"{FIELDS[field].label}, layer {layer}, t = {results.times[s]:g} d")
    if wells:
        _mark_wells(ax, results, layer,
                    (g.x_edges[:-1] + g.x_edges[1:]) / 2, (g.y_edges[:-1] + g.y_edges[1:]) / 2)
    if colorbar:
        (fig or ax.figure).colorbar(mesh, ax=ax, label=FIELDS[field].label)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax


def plot_layers(results: Results, field="sw", snapshot=-1, time=None, **kwargs):
    """All layers side by side on one colour scale."""
    g = results.grid
    lo, hi = color_limits(results, field, kwargs.pop("vmin", None), kwargs.pop("vmax", None))
    fig, axes = plt.subplots(1, g.nz, figsize=(4.2 * g.nz, 4.2), squeeze=False, layout="constrained")
    for k, ax in enumerate(axes[0], start=1):
        plot_layer(results, field, layer=k, snapshot=snapshot, time=time, ax=ax,
                   vmin=lo, vmax=hi, colorbar=False, title=f"Layer {k}", **kwargs)
    fig.suptitle(f"{FIELDS[field].label}, t = {results.times[pick_snapshot(results, snapshot, time)]:g} d")
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=kwargs.get("cmap") or FIELDS[field].cmap),
                 ax=axes.ravel().tolist(), label=FIELDS[field].label)
    return fig


def _draw_wells_in_section(ax, results, plane, index, along_edges):
    """Draw wellbores and perforations of the wells that lie in a vertical slice.

    A thin dashed line runs from the top of the model to the first perforation, a thick
    line covers the completed interval, and a square marks each perforated block.
    """
    z = results.grid.z_edges
    for w in well_completions(results):
        marker, color = _MARKERS[w["kind"]]
        in_slice = [c for c in w["cells"] if (c[1] if plane == "xz" else c[0]) == index]
        columns = {}
        for i, j, k in in_slice:
            columns.setdefault(i if plane == "xz" else j, []).append(k)
        for a, layers in columns.items():
            x_mid = (along_edges[a - 1] + along_edges[a]) / 2
            top, bottom = z[min(layers) - 1], z[max(layers)]
            ax.plot([x_mid, x_mid], [z[0], top], "--", color=color, lw=1.2, zorder=4)
            ax.plot([x_mid, x_mid], [top, bottom], "-", color=color, lw=4, zorder=4, solid_capstyle="butt")
            ax.plot([x_mid] * len(layers), [(z[k - 1] + z[k]) / 2 for k in layers], "s", color=color,
                    mec="white", ms=6, zorder=5)
            ax.annotate(w["name"], (x_mid, z[0]), xytext=(5, -3), textcoords="offset points",
                        fontsize=8, weight="bold", zorder=6)


def plot_cross_section(results: Results, field="sw", plane="xz", index=1, snapshot=-1, time=None,
                       ax=None, vmin=None, vmax=None, vertical_exaggeration=1.0, cmap=None,
                       colorbar=True, wells=True):
    """Vertical slice through the model.

    ``plane="xz"`` cuts along x at row ``index`` (counted from 1 in y);
    ``plane="yz"`` cuts along y at column ``index`` (counted from 1 in x).
    Depth increases downward.  Wells in the slice are drawn with their completed interval
    (thick line) and one square per perforated block.  ``vertical_exaggeration`` stretches the vertical axis,
    which helps because layers are usually thin compared with their length.
    """
    g = results.grid
    if plane not in ("xz", "yz"):
        raise ValueError("plane must be 'xz' or 'yz'")
    n_along, along_edges = (g.nx, g.x_edges) if plane == "xz" else (g.ny, g.y_edges)
    limit = g.ny if plane == "xz" else g.nx
    if not 1 <= index <= limit:
        raise ValueError(f"index must be between 1 and {limit} for plane {plane!r}")
    s = pick_snapshot(results, snapshot, time)
    lo, hi = color_limits(results, field, vmin, vmax)
    cube = g.to_3d(field_data(results, field)[s])            # (nz, ny, nx)
    section = cube[:, index - 1, :] if plane == "xz" else cube[:, :, index - 1]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    mesh = ax.pcolormesh(along_edges, g.z_edges, section, cmap=cmap or FIELDS[field].cmap,
                         vmin=lo, vmax=hi, edgecolors="0.7", linewidth=0.4)
    ax.invert_yaxis()
    if vertical_exaggeration != 1.0:
        ax.set_aspect(vertical_exaggeration, adjustable="box")
    if wells:
        _draw_wells_in_section(ax, results, plane, index, along_edges)
    where = f"row j={index}" if plane == "xz" else f"column i={index}"
    ax.set(xlabel=f"{plane[0]} (ft)", ylabel="depth (ft)",
           title=f"{FIELDS[field].label}, {plane} section at {where}, t = {results.times[s]:g} d")
    if colorbar:
        (fig or ax.figure).colorbar(mesh, ax=ax, label=FIELDS[field].label)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax


def _voxel_coordinates(g):
    """Corner coordinates for ``Axes3D.voxels``: arrays of shape (nx+1, ny+1, nz+1)."""
    return np.meshgrid(g.x_edges, g.y_edges, g.z_edges, indexing="ij")


def draw_wells_3d(ax, results: Results, stub=0.35):
    """Draw wellbores and perforations on a 3-D axes.

    Each well is a vertical line from above the top of the model (``stub`` times the model
    thickness) down through its perforated blocks, with a marker at the centre of every
    perforated block.  Producers are red, injectors blue.
    """
    g = results.grid
    cx, cy, cz = g.cell_centres()
    top = g.z_edges[0] - stub * (g.z_edges[-1] - g.z_edges[0])
    for w in well_completions(results):
        cells = [g.cell_index(b) for b in w["blocks"]]
        marker, color = _MARKERS[w["kind"]]
        xs, ys, zs = cx[cells], cy[cells], cz[cells]
        ax.plot([xs[0], xs[0]], [ys[0], ys[0]], [top, zs[0]], "--", color=color, lw=1.2, zorder=10)
        ax.plot(xs, ys, zs, "-", color=color, lw=3.5, zorder=10)
        ax.scatter(xs, ys, zs, marker="s", s=28, color=color, edgecolor="white", depthshade=False, zorder=11)
        ax.text(xs[0], ys[0], top, w["name"], color=color, fontsize=8, weight="bold", zorder=12)


def draw_voxels(ax, results: Results, field, snapshot, lo, hi, cmap=None, cutaway=False,
                vertical_exaggeration=3.0, alpha=0.9, wells=True):
    """Draw one snapshot as coloured 3-D blocks on an existing 3-D axes."""
    g = results.grid
    values = np.transpose(g.to_3d(field_data(results, field)[snapshot]), (2, 1, 0))   # (nx, ny, nz)
    filled = np.ones(values.shape, dtype=bool)
    if cutaway:                       # remove one corner so the inside is visible
        filled = np.transpose(g.to_3d(cutaway_keep(g)), (2, 1, 0)).copy()
    rgba = plt.get_cmap(cmap or FIELDS[field].cmap)(colors.Normalize(lo, hi)(values))
    rgba[..., 3] = alpha
    x, y, z = _voxel_coordinates(g)
    ax.computed_zorder = False          # draw the wells on top of the blocks, so they stay visible
    ax.voxels(x, y, z, filled, facecolors=rgba, edgecolor="0.35", linewidth=0.3, shade=False, zorder=1)
    if wells:
        draw_wells_3d(ax, results)
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_zlabel("depth (ft)")
    ax.set_box_aspect((g.x_edges[-1], g.y_edges[-1], (g.z_edges[-1] - g.z_edges[0]) * vertical_exaggeration))
    if not ax.zaxis_inverted():
        ax.invert_zaxis()


def plot_voxels(results: Results, field="sw", snapshot=-1, time=None, ax=None, vmin=None, vmax=None,
                cutaway=False, vertical_exaggeration=3.0, cmap=None, colorbar=True, wells=True):
    """The whole model as coloured 3-D blocks.  ``cutaway=True`` removes one corner to look inside.

    Wells are drawn on top of the blocks: a line through the completed layers with a square
    in every perforated block (producers red, injectors blue).
    """
    s = pick_snapshot(results, snapshot, time)
    lo, hi = color_limits(results, field, vmin, vmax)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(projection="3d")
    draw_voxels(ax, results, field, s, lo, hi, cmap, cutaway, vertical_exaggeration, wells=wells)
    ax.set_title(f"{FIELDS[field].label}, t = {results.times[s]:g} d")
    if colorbar:
        (fig or ax.figure).colorbar(
            plt.cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=cmap or FIELDS[field].cmap),
            ax=ax, shrink=0.6, pad=0.15, label=FIELDS[field].label)
    return fig if fig is not None else ax
