"""Time-lapse animations saved as GIF files (they need no extra software: only Pillow).

For an interactive 3-D time-lapse with a slider, see ``kusim.viz.interactive``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation, colors

from ..results import Results
from .fields import FIELDS, color_limits, frame_indices
from .maps import draw_voxels, plot_layer


def animate_layer(results: Results, path, field="sw", layer=1, fps=10, max_frames=60, **kwargs):
    """Save a GIF of one layer changing in time.  The colour scale is fixed over the whole run.

    Returns the path.  Extra keyword arguments go to ``plot_layer`` (``cmap``, ``wells``, ...).
    """
    path = Path(path)
    lo, hi = color_limits(results, field, kwargs.pop("vmin", None), kwargs.pop("vmax", None))
    frames = frame_indices(results, max_frames)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(lo, hi),
                                       cmap=kwargs.get("cmap") or FIELDS[field].cmap),
                 ax=ax, label=FIELDS[field].label)

    def draw(n):
        ax.clear()
        plot_layer(results, field, layer, snapshot=frames[n], ax=ax, vmin=lo, vmax=hi,
                   colorbar=False, **kwargs)

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def animate_voxels(results: Results, path, field="sw", fps=8, max_frames=40, cutaway=False,
                   vertical_exaggeration=3.0, cmap=None, azimuth=-60, elevation=25):
    """Save a GIF of the whole 3-D model changing in time (a fixed camera, a fixed colour scale)."""
    path = Path(path)
    lo, hi = color_limits(results, field)
    frames = frame_indices(results, max_frames)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=cmap or FIELDS[field].cmap),
                 ax=ax, shrink=0.6, pad=0.15, label=FIELDS[field].label)

    def draw(n):
        ax.clear()
        draw_voxels(ax, results, field, frames[n], lo, hi, cmap, cutaway, vertical_exaggeration)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(f"{FIELDS[field].label}, t = {results.times[frames[n]]:g} d")

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path
