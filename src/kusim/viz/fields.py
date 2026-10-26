"""Helpers shared by the plotting functions: which field, which snapshot, which colours."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..results import Results


@dataclass(frozen=True)
class Field:
    """How to show one cell property."""

    label: str
    cmap: str


FIELDS = {
    "pressure": Field("Oil pressure (psi)", "viridis"),
    "sw": Field("Water saturation", "Blues"),
    "so": Field("Oil saturation", "YlOrBr"),
}


def field_data(results: Results, name: str) -> np.ndarray:
    """All saved snapshots of a field as an array (n_snapshots, n_cells)."""
    if name == "pressure":
        return results.pressure
    if name == "sw":
        return results.sw
    if name == "so":
        return 1.0 - results.sw
    raise ValueError(f"unknown field {name!r}; choose from {sorted(FIELDS)}")


def pick_snapshot(results: Results, snapshot=-1, time=None) -> int:
    """Index of a saved snapshot, given an index or the time (days) nearest to ask for."""
    if time is not None:
        return int(np.argmin(np.abs(results.times - time)))
    n = len(results.times)
    if not -n <= snapshot < n:
        raise IndexError(f"snapshot {snapshot} is out of range for {n} saved snapshots")
    return snapshot % n


def color_limits(results: Results, name: str, vmin=None, vmax=None):
    """Colour range that stays fixed over all snapshots (so frames can be compared)."""
    data = field_data(results, name)
    lo = float(np.min(data)) if vmin is None else vmin
    hi = float(np.max(data)) if vmax is None else vmax
    if hi - lo < 1e-12:                    # a flat field (e.g. the first snapshot only)
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def frame_indices(results: Results, max_frames=60):
    """Evenly spaced snapshot indices (always including the first and the last)."""
    n = len(results.times)
    if n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_frames).round().astype(int).tolist()))


def well_completions(results: Results):
    """Every well with all of its completions.

    A list of dicts with ``name``, ``kind`` ("producer"/"injector"), ``blocks`` (thesis block
    numbers, from 1) and ``cells`` (one ``(i, j, k)`` per block, each from 1), sorted from the
    shallowest completion to the deepest.
    """
    if results.wells.empty:
        return []
    kinds = results.wells.drop_duplicates("well").set_index("well")["kind"]
    completions = results.completions or {
        name: [int(b)] for name, b in results.wells.drop_duplicates("well").set_index("well")["block"].items()
    }
    out = []
    for name, blocks in completions.items():
        cells = sorted((results.grid.block_ijk(b) for b in blocks), key=lambda c: c[2])
        out.append(dict(name=name, kind=kinds[name], blocks=[results.grid.block_number(*c) for c in cells],
                        cells=cells))
    return out


def cutaway_keep(grid):
    """Boolean mask of the cells to draw when a corner of the model is cut away.

    The cut removes the quarter with the largest x and the smallest y, so the
    inside of the model is visible from the usual viewing direction.
    """
    return ~((grid.i >= grid.nx // 2) & (grid.j < (grid.ny + 1) // 2))
