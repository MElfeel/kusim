"""Models, schedules, observations and results as plain JSON.

Everything an outside program (a script in another language, a web service, an LLM agent) needs to give to the
simulator, or to read from it, is a dictionary of numbers, strings, lists and dictionaries: no NumPy, no pandas.
The format is described in ``docs/json-format.md``.

    model = kusim.load_model("model.json")             # or ReservoirModel.from_dict(d)
    results = Simulator(model).run()
    kusim.save_results(results, "results.json")        # or results.to_dict()

Conventions
-----------
* Units are the oilfield units of the simulator (ft, md, psi, cp, STB/day, days).  Rates are positive.
* Block numbers count from 1 (the thesis numbering), x fastest, then y, then z (top layer first).
* ``None`` is written as ``null``.  A value that is not a number (for example the water cut of an injector) is
  written as ``null`` too, so the files are valid JSON.
* Reading is **strict**: an unknown key, a missing required key or a value of the wrong kind raises a
  ``ValueError`` that names the key, instead of being ignored.  A typo in a file is then an error, not a silent
  change of the physics.
"""

from __future__ import annotations

import difflib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .fluids import PVTTable, RockFluidTable
from .grid import Grid
from .model import Equilibrium, ReservoirModel, UniformStart
from .schedule import ControlEvent, Schedule
from .wells import Injector, Producer

SCHEMA_VERSION = 1
UNITS = "oilfield"

_MODEL_KEYS = {
    "schema_version", "units", "grid", "kx", "ky", "kz", "porosity", "rock_compressibility", "pvt", "rock_fluid",
    "gamma_o", "gamma_w", "swc", "sor", "initial", "dt", "t_end", "producers", "injectors", "provenance",
}
_MODEL_REQUIRED = _MODEL_KEYS - {"schema_version", "units", "producers", "injectors", "provenance"}
_GRID_KEYS = {"dx", "dy", "dz", "top_depth"}
_PVT_KEYS = {"pressure", "bo", "bw", "mu_o", "mu_w"}
_ROCK_FLUID_KEYS = {"sw", "kro", "krw", "pcow", "pc_multiplier"}
_PRODUCER_KEYS = {"block", "mode", "target", "radius", "skin", "min_bhp", "min_oil_rate", "wellbore_gravity"}
_INJECTOR_KEYS = {"block", "mode", "target", "radius", "skin", "wellbore_gravity"}
_EVENT_KEYS = {"time", "well", "mode", "target"}


# ---------------------------------------------------------------------- helpers
class SpecError(ValueError):
    """A problem with JSON input that can be located: ``path`` is where (``producers.0.min_bph``), ``code`` what
    (``unknown_key``, ``missing_key``, ``not_an_object``) and ``details`` carries hints (``allowed``, ``did_you_mean``).
    It is a ``ValueError``, so code that catches that keeps working."""

    def __init__(self, message, path="", code="invalid", **details):
        super().__init__(message)
        self.path, self.code, self.details = path, code, details

    def as_dict(self):
        return {"error": str(self), "code": self.code, "path": self.path, **self.details}


def _join(path, key):
    return f"{path}.{key}" if path else str(key)


def _check_keys(what, data, allowed, required=(), path=""):
    """Raise a SpecError (a ValueError) that names the unknown or missing keys of a dictionary."""
    if not isinstance(data, dict):
        raise SpecError(f"{what} must be a JSON object (a dictionary), got {type(data).__name__}", path, "not_an_object")
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        close = {k: difflib.get_close_matches(k, sorted(allowed), n=1) for k in unknown}
        raise SpecError(f"{what}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}",
                        _join(path, unknown[0]), "unknown_key", allowed=sorted(allowed),
                        did_you_mean={k: v[0] for k, v in close.items() if v})
    missing = sorted(set(required) - set(data))
    if missing:
        raise SpecError(f"{what}: missing required key(s) {missing}", _join(path, missing[0]), "missing_key",
                        missing=missing)


def _clean(value):
    """Make a value JSON-safe: NumPy types to Python, NaN and infinity to None, tuples to lists."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _compact(array):
    """One number when every value is the same, otherwise the list (for per-block data)."""
    array = np.asarray(array, dtype=float)
    return float(array.flat[0]) if np.all(array == array.flat[0]) else array.tolist()


def _frame_to_records(frame):
    return _clean(frame.to_dict("records")) if len(frame) else []


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise SpecError(f"{path} is not valid JSON: {error}", "", "invalid_json", line=error.lineno,
                        column=error.colno) from None


def _write_json(data, path, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=indent) + "\n", encoding="utf8")


# ---------------------------------------------------------------------- grid
def grid_to_dict(grid: Grid) -> dict:
    """The data needed to rebuild a ``Grid``: block sizes per column, row and layer, and the top depth."""
    top = grid.depth[: grid.n_layer] - grid.dz[0] / 2
    return {
        "dx": grid.dx[: grid.nx].tolist(),
        "dy": grid.dy[: grid.n_layer : grid.nx].tolist(),
        "dz": grid.dz[:: grid.n_layer].tolist(),
        "top_depth": _compact(top),
    }


def grid_from_dict(data) -> Grid:
    _check_keys("grid", data, _GRID_KEYS, required={"dx", "dy", "dz"}, path="grid")
    return Grid(data["dx"], data["dy"], data["dz"], top_depth=data.get("top_depth", 0.0))


# ---------------------------------------------------------------------- model
def _well_to_dict(well):
    d = {"block": well.blocks if len(well.blocks) > 1 else well.blocks[0], "mode": well.mode, "target": well.target,
         "radius": well.radius, "skin": well.skin}
    if isinstance(well, Producer):
        d["min_bhp"], d["min_oil_rate"] = well.min_bhp, well.min_oil_rate
    d["wellbore_gravity"] = well.wellbore_gravity
    return d


def _well_from_dict(kind, data, index):
    cls, keys = (Producer, _PRODUCER_KEYS) if kind == "producers" else (Injector, _INJECTOR_KEYS)
    _check_keys(f"a well in '{kind}'", data, keys, required={"block", "mode", "target"}, path=f"{kind}.{index}")
    return cls(**data)


def model_to_dict(model: ReservoirModel) -> dict:
    """A ``ReservoirModel`` as a JSON-ready dictionary (see ``docs/json-format.md``).

    The capillary-pressure column is written as the table the simulator uses (already multiplied by
    ``pc_multiplier``), with ``pc_multiplier`` equal to 1.
    """
    rf, init = model.rock_fluid, model.initial
    if isinstance(init, Equilibrium):
        initial = {"type": "equilibrium", "datum_pressure": init.datum_pressure, "woc_depth": init.woc_depth}
    else:
        initial = {"type": "uniform", "pressure": init.pressure, "sw": init.sw}
    return _clean({
        "schema_version": SCHEMA_VERSION,
        "units": UNITS,
        "grid": grid_to_dict(model.grid),
        "kx": _compact(model.kx), "ky": _compact(model.ky), "kz": _compact(model.kz),
        "porosity": _compact(model.porosity),
        "rock_compressibility": model.rock_compressibility,
        "pvt": {"pressure": model.pvt.pressure, "bo": model.pvt.bo, "bw": model.pvt.bw,
                "mu_o": model.pvt.mu_o, "mu_w": model.pvt.mu_w},
        "rock_fluid": {"sw": rf.sw, "kro": rf.kro, "krw": rf.krw, "pcow": rf.pcow, "pc_multiplier": 1.0},
        "gamma_o": model.gamma_o, "gamma_w": model.gamma_w, "swc": model.swc, "sor": model.sor,
        "initial": initial,
        "dt": model.dt, "t_end": model.t_end,
        "producers": [_well_to_dict(w) for w in model.producers],
        "injectors": [_well_to_dict(w) for w in model.injectors],
        "provenance": list(model.provenance),
    })


def model_from_dict(data) -> ReservoirModel:
    """Build a ``ReservoirModel`` from a dictionary written by ``model_to_dict`` (or by hand or by an agent).

    Raises ``ValueError`` for an unknown or missing key, or for data the model rejects (mismatched array lengths,
    a well in a block that does not exist, ...).
    """
    _check_keys("the model", data, _MODEL_KEYS, _MODEL_REQUIRED)
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise SpecError(f"schema_version {version} is not supported (this version of kusim reads {SCHEMA_VERSION})",
                        "schema_version", "unsupported_version")
    if data.get("units", UNITS) != UNITS:
        raise SpecError(f"units must be {UNITS!r}: kusim works in oilfield units only", "units", "unsupported_units")

    pvt, rock_fluid = data["pvt"], data["rock_fluid"]
    _check_keys("pvt", pvt, _PVT_KEYS, _PVT_KEYS, path="pvt")
    _check_keys("rock_fluid", rock_fluid, _ROCK_FLUID_KEYS, _ROCK_FLUID_KEYS - {"pc_multiplier"}, path="rock_fluid")

    init = dict(data["initial"]) if isinstance(data["initial"], dict) else data["initial"]
    _check_keys("initial", init, {"type", "datum_pressure", "woc_depth", "pressure", "sw"}, {"type"}, path="initial")
    kind = init.pop("type")
    if kind == "equilibrium":
        _check_keys("initial (equilibrium)", init, {"datum_pressure", "woc_depth"},
                    {"datum_pressure", "woc_depth"}, path="initial")
        initial = Equilibrium(**init)
    elif kind == "uniform":
        _check_keys("initial (uniform)", init, {"pressure", "sw"}, {"pressure", "sw"}, path="initial")
        initial = UniformStart(**init)
    else:
        raise SpecError(f"initial.type must be 'equilibrium' or 'uniform', got {kind!r}", "initial.type", "bad_value",
                        allowed=["equilibrium", "uniform"])

    return ReservoirModel(
        grid=grid_from_dict(data["grid"]),
        kx=data["kx"], ky=data["ky"], kz=data["kz"], porosity=data["porosity"],
        rock_compressibility=data["rock_compressibility"],
        pvt=PVTTable(**pvt),
        rock_fluid=RockFluidTable(**rock_fluid),
        gamma_o=data["gamma_o"], gamma_w=data["gamma_w"], swc=data["swc"], sor=data["sor"],
        initial=initial, dt=data["dt"], t_end=data["t_end"],
        producers=[_well_from_dict("producers", w, i) for i, w in enumerate(data.get("producers", []))],
        injectors=[_well_from_dict("injectors", w, i) for i, w in enumerate(data.get("injectors", []))],
        provenance=list(data.get("provenance", [])),
    )


def save_model(model: ReservoirModel, path):
    """Write a model to a JSON file."""
    _write_json(model_to_dict(model), path)


def load_model(path) -> ReservoirModel:
    """Read a model from a JSON file."""
    return model_from_dict(_read_json(path))


# ---------------------------------------------------------------------- schedule and observations
def schedule_to_list(schedule: Schedule) -> list:
    """A schedule as a list of ``{"time", "well", "mode", "target"}`` events."""
    return _clean([{"time": e.time, "well": e.well, "mode": e.mode, "target": e.target} for e in schedule.events])


def schedule_from_list(events) -> Schedule:
    """A ``Schedule`` from a list of ``{"time", "well", "mode"?, "target"?}`` dictionaries."""
    if not isinstance(events, list):
        raise ValueError("a schedule must be a JSON list of events")
    for i, event in enumerate(events):
        _check_keys("a schedule event", event, _EVENT_KEYS, {"time", "well"}, path=f"[{i}]")
    return Schedule.from_table(events)


def load_schedule(path) -> Schedule:
    return schedule_from_list(_read_json(path))


def save_schedule(schedule: Schedule, path):
    _write_json(schedule_to_list(schedule), path)


def observations_to_list(observations) -> list:
    """Observations as a list of ``{"time", "vector", "value", "sigma", "weight"}`` records."""
    return _frame_to_records(observations.table)


def observations_from_list(records):
    """``Observations`` from records with the keys time, vector, value, sigma and (optional) weight."""
    from .history import Observations

    if not isinstance(records, list):
        raise ValueError("observations must be a JSON list of records")
    for i, r in enumerate(records):
        _check_keys("an observation", r, {"time", "vector", "value", "sigma", "weight"},
                    {"time", "vector", "value", "sigma"}, path=f"[{i}]")
    return Observations(pd.DataFrame(records))


# ---------------------------------------------------------------------- results
FIELD_MODES = ("none", "last", "all")


def results_to_dict(results, fields="last") -> dict:
    """Results as a JSON-ready dictionary.

    ``fields`` says how much of the pressure and saturation maps to include: ``"none"`` (only the well tables),
    ``"last"`` (the final snapshot, the default) or ``"all"`` (every saved snapshot: large for big grids).
    The well tables (``wells``, ``perforations``) are lists of records, one per well per time step.
    """
    if fields not in FIELD_MODES:
        raise ValueError(f"fields must be one of {FIELD_MODES}, got {fields!r}")
    keep = {"none": slice(0, 0), "last": slice(-1, None), "all": slice(None)}[fields]
    return _clean({
        "schema_version": SCHEMA_VERSION,
        "units": UNITS,
        "grid": grid_to_dict(results.grid),
        "stop_reason": results.stop_reason,
        "completions": results.completions,
        "wells": _frame_to_records(results.wells),
        "perforations": _frame_to_records(results.perforations),
        "snapshot_times": results.times[keep],
        "pressure": results.pressure[keep],
        "sw": results.sw[keep],
    })


def results_from_dict(data):
    """Rebuild a ``Results`` from ``results_to_dict`` (with the snapshots that were included)."""
    from .results import Results

    keys = {"schema_version", "units", "grid", "stop_reason", "completions", "wells", "perforations",
            "snapshot_times", "pressure", "sw"}
    _check_keys("the results", data, keys, keys - {"schema_version", "units"})
    grid = grid_from_dict(data["grid"])
    n = grid.n_cells
    return Results(
        grid=grid,
        times=np.asarray(data["snapshot_times"], dtype=float),
        pressure=np.asarray(data["pressure"], dtype=float).reshape(-1, n),
        sw=np.asarray(data["sw"], dtype=float).reshape(-1, n),
        wells=pd.DataFrame(data["wells"]),
        perforations=pd.DataFrame(data["perforations"]),
        completions=data["completions"],
        stop_reason=data["stop_reason"],
    )


def save_results(results, path, fields="last"):
    """Write results to a JSON file (see ``results_to_dict`` for ``fields``)."""
    _write_json(results_to_dict(results, fields), path, indent=None)


def load_results(path):
    return results_from_dict(_read_json(path))


def run_summary(results, diagnostics=None) -> dict:
    from .simulator import diagnostics_healthy

    """A short, JSON-ready account of a run, for a person or an agent that does not want the full tables.

    For every well: its kind, control mode and the values at the last time step (bottom-hole pressure, rates,
    cumulative volumes).  ``diagnostics`` is ``Simulator.diagnostics()``; read it before trusting a run
    (pressure outside the PVT table, saturation clipping by more than 0.01 and backflow are signs of trouble).  ``healthy`` is
    true when none of those happened and no stop condition ended the run, false otherwise, and null when no
    diagnostics were given.
    """
    wells = {}
    if len(results.wells):
        last = results.wells[results.wells["time"] == results.wells["time"].max()]
        for row in last.to_dict("records"):
            name = row.pop("well")
            wells[name] = {k: v for k, v in row.items() if k != "time"}
    final_time = float(results.wells["time"].max()) if len(results.wells) else 0.0
    return _clean({
        "final_time": final_time,
        "n_cells": results.grid.n_cells,
        "stop_reason": results.stop_reason,
        "wells": wells,
        "diagnostics": diagnostics,
        "healthy": None if diagnostics is None else diagnostics_healthy(diagnostics),
    })


# ---------------------------------------------------------------------- restart state
_STATE_KEYS = {"schema_version", "units", "time", "pressure", "sw", "well_controls", "cum"}


def state_to_dict(state) -> dict:
    """A ``SimState`` as a JSON-ready dictionary: time, the pressure and saturation of every block, the control of
    every well (a producer that reached ``min_bhp`` is on ``"bhp"``) and the cumulative volumes (STB)."""
    return _clean({
        "schema_version": SCHEMA_VERSION, "units": UNITS,
        "time": state.time, "pressure": state.p_o, "sw": state.s_w,
        "well_controls": {name: {"mode": mode, "target": target} for name, (mode, target) in state.well_controls.items()},
        "cum": state.cum,
    })


def state_from_dict(data):
    """A ``SimState`` from ``state_to_dict`` output (strict). Whether it fits a model is checked by
    ``Simulator.set_state``."""
    from .simulator import SimState

    _check_keys("the state", data, _STATE_KEYS, _STATE_KEYS - {"schema_version", "units"})
    if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise SpecError(f"schema_version {data['schema_version']} is not supported (this version of kusim reads "
                        f"{SCHEMA_VERSION})", "schema_version", "unsupported_version")
    controls = {}
    for name, control in data["well_controls"].items():
        _check_keys(f"the control of well {name}", control, {"mode", "target"}, {"mode", "target"},
                    path=f"well_controls.{name}")
        controls[name] = (control["mode"], control["target"])
    return SimState(time=float(data["time"]), p_o=np.asarray(data["pressure"], dtype=float),
                    s_w=np.asarray(data["sw"], dtype=float), well_controls=controls,
                    cum={name: dict(volumes) for name, volumes in data["cum"].items()})


def save_state(state, path):
    """Write a restart state to a JSON file (floats are written in full: a restart reproduces the run exactly)."""
    _write_json(state_to_dict(state), path, indent=None)


def load_state(path):
    return state_from_dict(_read_json(path))


# ---------------------------------------------------------------------- parameter sets
_FIELD_PARAMETER_KEYS = {"type", "field", "transform", "lower", "upper", "ties", "name"}
_SCALAR_PARAMETER_KEYS = {"type", "path", "transform", "lower", "upper", "name"}


def parameters_to_dict(parameter_set) -> dict:
    """A ``ParameterSet`` as a JSON-ready dictionary (a list of field and scalar parameters)."""
    from .parameters import FieldParameter

    items = []
    for p in parameter_set.parameters:
        if isinstance(p, FieldParameter):
            items.append({"type": "field", "field": p.field, "transform": p.transform, "lower": p.lower,
                          "upper": p.upper, "ties": list(p.ties), "name": p.name})
        else:
            items.append({"type": "scalar", "path": p.path, "transform": p.transform, "lower": p.lower,
                          "upper": p.upper, "name": p.name})
    return _clean({"schema_version": SCHEMA_VERSION, "parameters": items})


def parameters_from_dict(data):
    """A ``ParameterSet`` from ``parameters_to_dict`` output (strict). ``type`` is ``"field"`` (one value per block:
    needs ``field``) or ``"scalar"`` (one number: needs ``path``); the other keys are optional."""
    from .parameters import FieldParameter, ParameterSet, ScalarParameter

    _check_keys("the parameters", data, {"schema_version", "parameters"}, {"parameters"})
    if not isinstance(data["parameters"], list):
        raise SpecError("parameters must be a list", "parameters", "not_a_list")
    built = []
    for i, item in enumerate(data["parameters"]):
        where = f"parameters.{i}"
        _check_keys("a parameter", item, _FIELD_PARAMETER_KEYS | _SCALAR_PARAMETER_KEYS, {"type"}, path=where)
        kind = item["type"]
        if kind == "field":
            _check_keys("a field parameter", item, _FIELD_PARAMETER_KEYS, {"type", "field"}, path=where)
            built.append(FieldParameter(**{k: v for k, v in item.items() if k != "type"}))
        elif kind == "scalar":
            _check_keys("a scalar parameter", item, _SCALAR_PARAMETER_KEYS, {"type", "path"}, path=where)
            built.append(ScalarParameter(**{k: v for k, v in item.items() if k != "type"}))
        else:
            raise SpecError(f"a parameter's type must be 'field' or 'scalar', got {kind!r}", f"{where}.type",
                            "bad_value", allowed=["field", "scalar"])
    return ParameterSet(built)


def load_observations(path):
    return observations_from_list(_read_json(path))


def save_observations(observations, path):
    _write_json(observations_to_list(observations), path)


# ---------------------------------------------------------------------- controls as optimisation variables
_CONTROL_KEYS = {"well", "times", "initial", "mode", "lower", "upper", "name"}


def controls_to_dict(control_set, initial: dict) -> dict:
    """A ``ControlSet`` with the starting target of every period (``initial = {well: [target of period 0, 1, ...]}``)
    as a JSON-ready dictionary."""
    items = []
    for p in control_set.parameters:
        items.append({"well": p.well, "times": list(p.times), "initial": [float(v) for v in initial[p.well]],
                      "mode": p.mode, "lower": p.lower, "upper": p.upper, "name": p.name})
    return _clean({"schema_version": SCHEMA_VERSION, "controls": items})


def controls_from_dict(data):
    """``(ControlSet, initial)`` from ``controls_to_dict`` output (strict). Per well: ``well`` (``P1``, ``I1``),
    ``times`` (start day of each control period, increasing), ``initial`` (one starting target per period) and
    optionally ``mode``, ``lower`` (default 0) and ``upper`` bounds."""
    from .schedule import ControlParameter, ControlSet

    _check_keys("the controls", data, {"schema_version", "controls"}, {"controls"})
    if not isinstance(data["controls"], list):
        raise SpecError("controls must be a list", "controls", "not_a_list")
    parameters, initial = [], {}
    for i, item in enumerate(data["controls"]):
        where = f"controls.{i}"
        _check_keys("a control", item, _CONTROL_KEYS, {"well", "times", "initial"}, path=where)
        if len(item["initial"]) != len(item["times"]):
            raise SpecError(f"{item['well']}: initial needs one target per period ({len(item['times'])}), "
                            f"got {len(item['initial'])}", f"{where}.initial", "bad_value")
        initial[item["well"]] = list(item["initial"])
        parameters.append(ControlParameter(**{k: v for k, v in item.items() if k != "initial"}))
    return ControlSet(parameters), initial


# ---------------------------------------------------------------------- prior of an ensemble
_PRIOR_ENTRY_KEYS = {"mean", "mean_by_layer", "std", "range_xy", "range_z"}


def prior_to_dict(prior: dict) -> dict:
    """A prior (``{parameter name: {"std": ..., "mean"?, "mean_by_layer"?, "range_xy"?, "range_z"?}}``) as JSON."""
    return _clean({"schema_version": SCHEMA_VERSION, "parameters": prior})


def prior_from_dict(data) -> dict:
    """The prior of an ensemble from JSON (strict): ``{"parameters": {name: entry}}`` where ``name`` is a parameter of
    the ``ParameterSet`` (``log_kx``, ``initial.woc_depth``, ...) and an entry has ``std`` (required, in the transformed
    space: ln units for a log parameter) and optionally ``mean`` (physical value, one number or one per block) or
    ``mean_by_layer`` (physical, one per layer), and for a field ``range_xy`` and ``range_z`` (ft; exponential
    correlation). Without a mean the value in the model is the mean. Whether the names fit the parameter set is checked
    by ``kusim.ensemble.sample_prior``."""
    _check_keys("the prior", data, {"schema_version", "parameters"}, {"parameters"})
    entries = data["parameters"]
    if not isinstance(entries, dict) or not entries:
        raise SpecError("the prior needs a non-empty object of parameter entries", "parameters", "not_an_object")
    for name, entry in entries.items():
        _check_keys(f"the prior of {name}", entry, _PRIOR_ENTRY_KEYS, {"std"}, path=f"parameters.{name}")
    return {name: dict(entry) for name, entry in entries.items()}


def load_prior(path) -> dict:
    return prior_from_dict(_read_json(path))
