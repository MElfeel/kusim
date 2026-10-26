"""Simulation results: field snapshots and well histories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .grid import Grid


@dataclass
class Results:
    """What a run produces.

    Attributes
    ----------
    grid : Grid
    times : array
        Time (days) of each saved snapshot, starting with 0.
    pressure, sw : arrays of shape (n_snapshots, n_cells)
        Oil pressure (psi) and water saturation at each saved time.
    wells : DataFrame
        One row per well per time step: time, well, kind, block (the first completion),
        n_perforations, bhp, oil_rate, water_rate, liquid_rate, water_cut and cumulative volumes.
    perforations : DataFrame
        One row per perforation per time step: time, well, kind, block, i, j, k, oil_rate,
        water_rate and the pressure of the block.  Shows how a multi-layer well draws from
        (or injects into) each layer.
    completions : dict
        Block numbers of each well's completions, e.g. ``{"P1": [10, 35, 60]}``.
    stop_reason : str or None
        Why the run ended early, if it did.
    """

    grid: Grid
    times: np.ndarray
    pressure: np.ndarray
    sw: np.ndarray
    wells: pd.DataFrame
    perforations: pd.DataFrame = field(default_factory=pd.DataFrame)
    completions: dict = field(default_factory=dict)
    stop_reason: str | None = None

    def pressure_3d(self, snapshot=-1):
        """Pressure at one saved time as an array of shape (nz, ny, nx)."""
        return self.grid.to_3d(self.pressure[snapshot])

    def sw_3d(self, snapshot=-1):
        """Water saturation at one saved time as an array of shape (nz, ny, nx)."""
        return self.grid.to_3d(self.sw[snapshot])

    def well(self, name):
        """History of one well, e.g. ``results.well("P1")``."""
        return self.wells[self.wells["well"] == name].reset_index(drop=True)

    def perforations_of(self, name):
        """Per-perforation history of one well (one row per perforation per step)."""
        return self.perforations[self.perforations["well"] == name].reset_index(drop=True)

    def to_dict(self, fields="last") -> dict:
        """The results as a JSON-ready dictionary; ``fields`` is "none", "last" or "all" (see ``kusim.spec``)."""
        from .spec import results_to_dict

        return results_to_dict(self, fields)

    def to_json(self, path, fields="last"):
        """Write the results to a JSON file (the well tables and, depending on ``fields``, the maps)."""
        from .spec import save_results

        save_results(self, path, fields)

    @classmethod
    def from_json(cls, path) -> "Results":
        """Read results written by ``to_json``."""
        from .spec import load_results

        return load_results(path)

    def to_csv(self, folder):
        """Write ``wells.csv``, ``pressure.csv`` and ``sw.csv`` into ``folder``."""
        from pathlib import Path

        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        self.wells.to_csv(out / "wells.csv", index=False)
        self.perforations.to_csv(out / "perforations.csv", index=False)
        columns = [f"block_{n}" for n in range(1, self.grid.n_cells + 1)]
        for name, data in (("pressure", self.pressure), ("sw", self.sw)):
            frame = pd.DataFrame(data, columns=columns)
            frame.insert(0, "time", self.times)
            frame.to_csv(out / f"{name}.csv", index=False)
