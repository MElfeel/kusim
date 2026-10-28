"""KUSIM: a 3-D two-phase (oil/water) IMPES reservoir simulator for teaching."""

from importlib.metadata import PackageNotFoundError, version as _version

from .fluids import PVTTable, RockFluidTable
from .grid import Grid
from .io import read_legacy_input
from .model import Equilibrium, ReservoirModel, UniformStart
from .results import Results
from .simulator import SimState, Simulator
from .spec import (load_model, load_observations, load_results, load_schedule, load_state, save_model, save_observations,
                   save_results, save_schedule, save_state)
from .wells import Injector, Producer

try:
    __version__ = _version("kusim")
except PackageNotFoundError:        # running from a source tree that was not installed
    __version__ = "0+unknown"

__all__ = [
    "Equilibrium",
    "Grid",
    "Injector",
    "PVTTable",
    "Producer",
    "ReservoirModel",
    "Results",
    "RockFluidTable",
    "SimState",
    "Simulator",
    "UniformStart",
    "__version__",
    "load_model",
    "load_observations",
    "load_results",
    "load_schedule",
    "load_state",
    "read_legacy_input",
    "save_model",
    "save_observations",
    "save_results",
    "save_schedule",
    "save_state",
]
