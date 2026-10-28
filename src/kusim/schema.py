"""JSON Schema (draft 2020-12) of the kusim files, for tools and agents: ``kusim schema model``.

The schemas are written by hand next to the readers in ``spec.py``; ``tests/test_schema.py`` checks that they list
exactly the keys the readers accept, and that the example model validates, so the two cannot drift apart. The schema
covers the shape (keys, types, allowed values). Rules that need several fields at once (array lengths, block
numbers inside the grid, schedule times that are multiples of ``dt``) are checked by ``kusim validate``.
"""

from __future__ import annotations

from .spec import SCHEMA_VERSION, UNITS
from .wells import INJECTOR_MODES, PRODUCER_MODES

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_NUM = {"type": "number"}
_NUMS = {"type": "array", "items": _NUM}
_NUM_OR_NULL = {"type": ["number", "null"]}
_BLOCK = {"type": "integer", "minimum": 1, "description": "block number, counted from 1 (x fastest, then y, then z)"}
_PER_BLOCK = {"oneOf": [_NUM, _NUMS]}
_PER_BLOCK_NOTE = "one number, or one number per block (nx*ny*nz, x fastest)"


def _obj(properties, required=(), description=None):
    out = {"type": "object", "properties": properties, "required": sorted(required), "additionalProperties": False}
    if description:
        out["description"] = description
    return out


def _well(modes, producer):
    props = {
        "block": {"description": "a block number, or a list of block numbers (a well completed in several layers)",
                  "oneOf": [_BLOCK, {"type": "array", "items": _BLOCK, "minItems": 1}]},
        "mode": {"enum": list(modes)},
        "target": {**_NUM, "description": "STB/day for the rate modes, psi for 'bhp'"},
        "radius": {**_NUM, "exclusiveMinimum": 0, "description": "wellbore radius, ft"},
        "skin": _NUM,
        "wellbore_gravity": {"type": ["boolean", "null"]},
    }
    if producer:
        props["min_bhp"] = {**_NUM_OR_NULL, "description": "psi; the well switches to this pressure when reached"}
        props["min_oil_rate"] = {**_NUM_OR_NULL, "description": "STB/day; the run stops below this rate"}
    return _obj(props, {"block", "mode", "target"})


def _grid_schema():
    return _obj({
        "dx": {**_NUMS, "minItems": 1, "description": "block size in x, one per column (nx), ft"},
        "dy": {**_NUMS, "minItems": 1, "description": "block size in y, one per row (ny), ft"},
        "dz": {**_NUMS, "minItems": 1, "description": "block size in z, one per layer (nz), ft"},
        "top_depth": {**_PER_BLOCK, "description": "depth of the top of layer 1, ft: one number or one per column "
                                                   "of blocks (nx*ny)"},
    }, {"dx", "dy", "dz"})


def model_schema():
    per_block = lambda text: {**_PER_BLOCK, "description": f"{text}; {_PER_BLOCK_NOTE}"}
    return {
        "$schema": _DRAFT,
        "title": "kusim model",
        **_obj({
            "schema_version": {"const": SCHEMA_VERSION},
            "units": {"const": UNITS, "description": "ft, md, psi, cp, STB/day, days, 1/psi"},
            "grid": _grid_schema(),
            "kx": per_block("permeability in x, md"),
            "ky": per_block("permeability in y, md"),
            "kz": per_block("permeability in z, md"),
            "porosity": per_block("fraction"),
            "rock_compressibility": {**_NUM, "description": "1/psi"},
            "pvt": _obj({"pressure": {**_NUMS, "description": "psi, increasing"}, "bo": _NUMS, "bw": _NUMS,
                         "mu_o": {**_NUMS, "description": "cp"}, "mu_w": {**_NUMS, "description": "cp"}},
                        {"pressure", "bo", "bw", "mu_o", "mu_w"}),
            "rock_fluid": _obj({"sw": {**_NUMS, "description": "increasing"}, "kro": _NUMS, "krw": _NUMS,
                                "pcow": {**_NUMS, "description": "psi"},
                                "pc_multiplier": _NUM}, {"sw", "kro", "krw", "pcow"}),
            "gamma_o": {**_NUM, "description": "oil specific gravity (water = 1)"},
            "gamma_w": {**_NUM, "description": "water specific gravity"},
            "swc": {**_NUM, "description": "connate water saturation"},
            "sor": {**_NUM, "description": "residual oil saturation"},
            "initial": {"oneOf": [
                _obj({"type": {"const": "equilibrium"}, "datum_pressure": _NUM, "woc_depth": _NUM},
                     {"type", "datum_pressure", "woc_depth"}, "capillary-gravity equilibrium; needs a pcow table"),
                _obj({"type": {"const": "uniform"}, "pressure": _NUM, "sw": _NUM}, {"type", "pressure", "sw"}),
            ]},
            "dt": {**_NUM, "exclusiveMinimum": 0, "description": "time step, days"},
            "t_end": {**_NUM, "exclusiveMinimum": 0, "description": "simulated time, days"},
            "producers": {"type": "array", "items": _well(PRODUCER_MODES, True),
                          "description": "named P1, P2, ... in this order"},
            "injectors": {"type": "array", "items": _well(INJECTOR_MODES, False),
                          "description": "named I1, I2, ... in this order"},
            "provenance": {"type": "array", "items": {"type": "string"}},
        }, {"grid", "kx", "ky", "kz", "porosity", "rock_compressibility", "pvt", "rock_fluid", "gamma_o", "gamma_w",
            "swc", "sor", "initial", "dt", "t_end"}),
    }


def schedule_schema():
    return {
        "$schema": _DRAFT,
        "title": "kusim schedule",
        "type": "array",
        "items": _obj({
            "time": {**_NUM, "minimum": 0, "description": "days; a multiple of the model's dt"},
            "well": {"type": "string", "pattern": "^[PI][0-9]+$", "description": "P1.. or I1.."},
            "mode": {"enum": sorted(set(PRODUCER_MODES) | set(INJECTOR_MODES)),
                     "description": "optional; must suit the kind of well"},
            "target": {**_NUM, "description": "optional; at least one of mode and target is needed"},
        }, {"time", "well"}),
    }


def observations_schema():
    return {
        "$schema": _DRAFT,
        "title": "kusim observations",
        "type": "array",
        "items": _obj({
            "time": {**_NUM, "description": "days"},
            "vector": {"type": "string", "description": "summary vector such as WBHP:P1 or WOPR:P1"},
            "value": _NUM,
            "sigma": {**_NUM, "exclusiveMinimum": 0, "description": "standard deviation of the error"},
            "weight": _NUM,
        }, {"time", "vector", "value", "sigma"}),
    }


def results_schema():
    table = {"type": "array", "items": {"type": "object"}, "description": "one record per well per time step"}
    return {
        "$schema": _DRAFT,
        "title": "kusim results",
        **_obj({
            "schema_version": {"const": SCHEMA_VERSION}, "units": {"const": UNITS},
            "grid": _grid_schema(),
            "stop_reason": {"type": ["string", "null"]},
            "completions": {"type": "object"},
            "wells": table, "perforations": table,
            "snapshot_times": _NUMS,
            "pressure": {"type": "array", "items": _NUMS, "description": "psi; one map of nx*ny*nz values per snapshot"},
            "sw": {"type": "array", "items": _NUMS},
        }, {"grid", "stop_reason", "completions", "wells", "perforations", "snapshot_times", "pressure", "sw"}),
    }


SCHEMAS = {"model": model_schema, "schedule": schedule_schema, "observations": observations_schema,
           "results": results_schema}


def get_schema(name):
    if name not in SCHEMAS:
        raise ValueError(f"unknown schema {name!r}; choose one of {sorted(SCHEMAS)}")
    return SCHEMAS[name]()
