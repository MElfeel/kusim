"""The reservoir model: everything the simulator needs, gathered in one object.

A ``ReservoirModel`` is plain data.  Build one in Python (see the examples) or
read one from the original ``INPUT.DAT`` format with ``kusim.io.read_legacy_input``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from .fluids import PVTTable, RockFluidTable
from .grid import Grid
from .wells import Injector, Producer


@dataclass
class Equilibrium:
    """Initial state from capillary-gravity equilibrium (the original program's method).

    Pressure follows the oil gradient from the datum, and water saturation comes
    from the capillary pressure at each block's height above the water-oil contact:

        Pcow = 0.433 * (gamma_w - gamma_o) * (woc_depth - depth)
    """

    datum_pressure: float   # psi, at depth 0 (the reference datum)
    woc_depth: float        # ft below the datum


@dataclass
class UniformStart:
    """Initial state with the same pressure and water saturation in every block."""

    pressure: float
    sw: float


@dataclass
class ReservoirModel:
    """All data for one simulation run.

    Attributes
    ----------
    grid : Grid
    kx, ky, kz : float or array
        Permeability (md): one number or one value per block.
    porosity : float or array
    rock_compressibility : float
        cr, 1/psi.
    pvt : PVTTable
    rock_fluid : RockFluidTable
    gamma_o, gamma_w : float
        Oil and water specific gravity (water = 1).
    swc, sor : float
        Connate water and residual oil saturation; Sw is kept in [swc, 1 - sor].
    initial : Equilibrium or UniformStart
    producers, injectors : list
    dt, t_end : float
        Time step and end time, days.
    provenance : list of str
        What was changed since the model was built (filled by ``with_changes``).

    Edit a model with ``with_changes`` (it returns a new model and leaves this one alone).
    """

    grid: Grid
    kx: Union[float, np.ndarray]
    ky: Union[float, np.ndarray]
    kz: Union[float, np.ndarray]
    porosity: Union[float, np.ndarray]
    rock_compressibility: float
    pvt: PVTTable
    rock_fluid: RockFluidTable
    gamma_o: float
    gamma_w: float
    swc: float
    sor: float
    initial: Union[Equilibrium, UniformStart]
    dt: float
    t_end: float
    producers: List[Producer] = field(default_factory=list)
    injectors: List[Injector] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)

    def __post_init__(self):
        g = self.grid
        self.kx = g.as_cell_array(self.kx, "kx")
        self.ky = g.as_cell_array(self.ky, "ky")
        self.kz = g.as_cell_array(self.kz, "kz")
        self.porosity = g.as_cell_array(self.porosity, "porosity")
        if self.dt <= 0 or self.t_end < 0:
            raise ValueError("dt must be positive and t_end must not be negative")
        for well in [*self.producers, *self.injectors]:
            for block in well.blocks:
                g.cell_index(block)  # raises if the block does not exist
        blocks = [b for w in [*self.producers, *self.injectors] for b in w.blocks]
        if len(blocks) != len(set(blocks)):
            raise ValueError("two wells (or two completions) share a block, which is not supported")

    # ------------------------------------------------------------------
    # Editing (history matching, ensembles)
    # ------------------------------------------------------------------
    def with_changes(self, changes=None, note=None, **fields):
        """A new model with some data replaced.  This model is not changed.

        ``changes`` is a dictionary and/or keyword arguments.  A key is a model attribute (``"kx"``,
        ``"porosity"``, ``"rock_compressibility"``) or a path into it: ``"initial.woc_depth"``,
        ``"producers[0].skin"``, ``"injectors[0].target"``.  Arrays may be given as one number or one value per
        block, as in the constructor.  The new model is a deep copy (nothing is shared with the old one) and is
        validated again; ``note`` (or the list of changed keys) is appended to ``provenance``.

        >>> better = model.with_changes(kx=model.kx * 1.2, note="layer permeability +20 %")
        """
        changes = {**(changes or {}), **fields}
        new = copy.deepcopy(self)
        for path, value in changes.items():
            _set_path(new, path, value)
        new.__post_init__()
        new.provenance = list(self.provenance) + [note or "changed " + ", ".join(sorted(changes))]
        return new

    def get(self, path):
        """Read an attribute or a path (see ``with_changes``)."""
        return _get_path(self, path)

    # ------------------------------------------------------------------
    # JSON (see kusim.spec and docs/json-format.md)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """The model as a JSON-ready dictionary."""
        from .spec import model_to_dict

        return model_to_dict(self)

    @classmethod
    def from_dict(cls, data) -> "ReservoirModel":
        """A model from a dictionary written by ``to_dict`` (strict: unknown or missing keys raise ValueError)."""
        from .spec import model_from_dict

        return model_from_dict(data)

    def to_json(self, path):
        """Write the model to a JSON file."""
        from .spec import save_model

        save_model(self, path)

    @classmethod
    def from_json(cls, path) -> "ReservoirModel":
        """Read a model from a JSON file."""
        from .spec import load_model

        return load_model(path)

    @property
    def n_steps(self) -> int:
        """Number of time steps to reach ``t_end``."""
        return int(round(self.t_end / self.dt))


# ---------------------------------------------------------------------- attribute paths
_STEP = re.compile(r"^([A-Za-z_]\w*)(?:\[(\d+)\])?$")


def _steps(path):
    """'producers[0].skin' -> [('producers', 0), ('skin', None)]"""
    out = []
    for part in path.split("."):
        m = _STEP.match(part)
        if m is None:
            raise ValueError(f"cannot read {path!r}: {part!r} is not an attribute name or name[index]")
        out.append((m.group(1), None if m.group(2) is None else int(m.group(2))))
    return out


def _get_path(obj, path):
    for name, index in _steps(path):
        if not hasattr(obj, name):
            raise AttributeError(f"{type(obj).__name__} has no attribute {name!r} (path {path!r})")
        obj = getattr(obj, name)
        if index is not None:
            obj = obj[index]
    return obj


def _set_path(obj, path, value):
    steps = _steps(path)
    for name, index in steps[:-1]:
        obj = getattr(obj, name)
        if index is not None:
            obj = obj[index]
    name, index = steps[-1]
    if not hasattr(obj, name):
        raise AttributeError(f"{type(obj).__name__} has no attribute {name!r} (path {path!r})")
    if index is None:
        setattr(obj, name, value)
    else:
        getattr(obj, name)[index] = value
