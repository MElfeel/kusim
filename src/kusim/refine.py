"""Refine the grid of a model: every block is cut into ``fx * fy * fz`` smaller ones.

"Double the number of cells in x, y and z" is a request a student or an agent makes often (a grid-sensitivity study), and it is
easy to get wrong by hand: the block sizes, the per-block properties and the **block numbers of the wells** all change.
``refine_model`` does it in one step and says what it did.

What is kept and what is decided:

* the block sizes are divided (``dx`` by ``fx``, ...), so the reservoir has the same size; the total pore volume is unchanged;
* a property given per block (permeability, porosity) is copied to the children of its block (no interpolation);
* a well keeps its place: it is completed in the child column nearest to the centre of its old block (for an even factor the
  two middle children are equally near; the one with the greater index is taken) and in **all** children of the old layer,
  so its vertical extent is the same. A well with several completions keeps every one;
* a dipping top depth (one value per column) is copied to the child columns (a staircase, not a smooth surface);
* the well index (Peaceman) is computed again from the new block sizes, so a pressure-controlled well does not give exactly
  the old rate (that is the discretisation error you are measuring);
* the time step is not scaled with the blocks: the explicit saturation update needs a smaller ``dt`` on smaller blocks, so
  ``refine_model(..., dt="auto")`` runs the stability probe of ``kusim.checks`` and uses its suggested ``dt/k`` when the old
  ``dt`` would oscillate (the answer says so).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .grid import Grid
from .model import ReservoirModel


def _split(sizes, factor):
    """Every size cut into ``factor`` equal parts (a list ``factor`` times longer)."""
    return np.repeat(np.asarray(sizes, dtype=float) / factor, factor)


def _refine_cells(grid, array, fx, fy, fz):
    """A per-block array on the refined grid: each value is copied to the children of its block."""
    cube = grid.to_3d(array)
    return np.repeat(np.repeat(np.repeat(cube, fz, axis=0), fy, axis=1), fx, axis=2).ravel()


def _refine_blocks(old, new, blocks, fx, fy, fz):
    """Block numbers on the refined grid for the blocks of a well (see the module docstring)."""
    out = []
    for block in blocks:
        i, j, k = old.block_ijk(block)
        ci, cj = (i - 1) * fx + fx // 2 + 1, (j - 1) * fy + fy // 2 + 1
        for ck in range((k - 1) * fz + 1, k * fz + 1):
            number = new.block_number(ci, cj, ck)
            if number not in out:
                out.append(number)
    return out


def refine_grid_and_wells(model, fx, fy, fz):
    """The refined model (same ``dt``); see ``refine_model``."""
    old = model.grid
    top = old.depth[: old.n_layer] - old.dz[0] / 2
    top_depth = np.repeat(top.reshape(old.ny, old.nx), fy, axis=0)
    top_depth = np.repeat(top_depth, fx, axis=1).ravel()
    new = Grid(_split(old.dx[: old.nx], fx), _split(old.dy[: old.n_layer : old.nx], fy), _split(old.dz[:: old.n_layer], fz),
               top_depth=float(top_depth[0]) if np.all(top_depth == top_depth[0]) else top_depth)

    def well(w):
        return dataclasses.replace(w, block=_refine_blocks(old, new, w.blocks, fx, fy, fz))

    return dataclasses.replace(
        model, grid=new,
        kx=_refine_cells(old, model.kx, fx, fy, fz), ky=_refine_cells(old, model.ky, fx, fy, fz),
        kz=_refine_cells(old, model.kz, fx, fy, fz), porosity=_refine_cells(old, model.porosity, fx, fy, fz),
        producers=[well(w) for w in model.producers], injectors=[well(w) for w in model.injectors],
        provenance=list(model.provenance) + [f"refined the grid by {fx} x {fy} x {fz} (x, y, z)"])


def refine_model(model: ReservoirModel, fx=2, fy=None, fz=None, dt="auto", upstream="potential"):
    """``(new_model, info)``: ``model`` with every block cut into ``fx`` parts in x, ``fy`` in y, ``fz`` in z.

    ``fy`` and ``fz`` default to ``fx``. ``dt`` is ``"auto"`` (keep the time step unless the stability probe says it is too large,
    then use the suggested ``dt/k``), ``"keep"``, or a number of days (a divisor of the old ``dt`` keeps schedule and
    observation times valid). ``info`` describes the change: old and new shape and cell count, where every well went, and the
    time step decision.
    """
    fy = fx if fy is None else fy
    fz = fx if fz is None else fz
    for name, value in (("fx", fx), ("fy", fy), ("fz", fz)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"the refinement factor {name} must be a whole number of at least 1, got {value!r}")
    new = refine_grid_and_wells(model, int(fx), int(fy), int(fz))
    info = {
        "factors_xyz": [int(fx), int(fy), int(fz)],
        "old_shape_nx_ny_nz": [model.grid.nx, model.grid.ny, model.grid.nz], "old_n_cells": model.grid.n_cells,
        "new_shape_nx_ny_nz": [new.grid.nx, new.grid.ny, new.grid.nz], "new_n_cells": new.grid.n_cells,
        "wells": {f"{'P' if kind == 'producers' else 'I'}{n}": {"old_blocks": a.blocks, "new_blocks": b.blocks}
                  for kind in ("producers", "injectors")
                  for n, (a, b) in enumerate(zip(getattr(model, kind), getattr(new, kind)), start=1)},
    }
    if dt == "keep":
        info["dt_decision"] = {"decision": "kept", "old": model.dt, "new": model.dt}
    elif dt == "auto":
        new = auto_dt(new, info, upstream)
    else:
        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        new = dataclasses.replace(new, dt=dt)
        info["dt_decision"] = {"decision": "set by the caller", "old": model.dt, "new": dt}
    return new, info


def auto_dt(new, info, upstream="potential"):
    """Apply the stability probe to a refined model: returns the model with the suggested ``dt/k`` if the old step is too large.

    Also fills ``info["dt_decision"]``, ``info["findings"]`` and ``info["estimate"]`` (the findings of ``check_model`` for the model that is returned)."""
    from .checks import check_model

    check = check_model(new, probe_steps=3, max_cells=0, max_seconds=0, max_output_mb=0, upstream=upstream)
    info["dt_decision"] = {"decision": "kept", "old": new.dt, "new": new.dt}
    suggested = check["stability"].get("suggested_dt")
    if suggested and suggested < new.dt:
        info["dt_decision"] = {"decision": "reduced: the old dt is likely to oscillate on the smaller blocks", "old": new.dt, "new": suggested}
        new = dataclasses.replace(new, dt=suggested)
        check = check_model(new, probe_steps=3, max_cells=0, max_seconds=0, max_output_mb=0, upstream=upstream)
    info["findings"] = check["findings"]
    info["estimate"] = check["estimate"]
    return new
