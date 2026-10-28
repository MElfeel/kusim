# JSON format and command line

Everything that goes into the simulator, and the results that come out, can be plain JSON. This is for scripts in
other languages, services, and LLM agents. Code: `src/kusim/spec.py` (the format) and `src/kusim/cli.py` (the
command line). The format version is `schema_version: 1`.

## Rules

- **Units are oilfield units**: ft, md, psi, cp, STB/day, days, 1/psi. A file with `"units"` other than
  `"oilfield"` is rejected. Rates are positive for producers and injectors.
- **Block numbers count from 1**: x fastest, then y, then z, top layer first (`block = (k-1)*nx*ny + (j-1)*nx + i`).
- **Wells are named by order**: producers `P1, P2, ...` and injectors `I1, I2, ...` (the names used in schedules,
  summary vectors and the results tables).
- **Reading is strict.** An unknown key, a missing required key or a bad value raises a `ValueError` (or, on the
  command line, `{"ok": false, "error": ...}`) that names it. A typo such as `min_bph` is an error, not a silent
  change of the physics.
- `null` is `None`. A value that is not a number (the water cut of an injector) is written as `null`; files never
  contain `NaN`.
- Per-block data (`kx`, `ky`, `kz`, `porosity`) are one number or a list of `nx*ny*nz` numbers. Writing a model
  folds uniform arrays to one number.

## Model

```json
{
 "schema_version": 1,
 "units": "oilfield",
 "grid": {"dx": [50, 50], "dy": [50], "dz": [20, 20], "top_depth": 0.0},
 "kx": 100.0, "ky": 100.0, "kz": 10.0,
 "porosity": 0.2,
 "rock_compressibility": 4e-6,
 "pvt": {"pressure": [1000, 5000], "bo": [1.3, 1.2], "bw": [1.02, 1.01], "mu_o": [1.5, 1.5], "mu_w": [0.5, 0.5]},
 "rock_fluid": {"sw": [0.2, 0.8], "kro": [1, 0], "krw": [0, 0.4], "pcow": [0, 0], "pc_multiplier": 1.0},
 "gamma_o": 0.85, "gamma_w": 1.0, "swc": 0.2, "sor": 0.2,
 "initial": {"type": "uniform", "pressure": 3000.0, "sw": 0.2},
 "dt": 1.0, "t_end": 100.0,
 "producers": [{"block": 4, "mode": "bhp", "target": 2500.0, "radius": 0.25, "skin": 0.0,
                "min_bhp": null, "min_oil_rate": null, "wellbore_gravity": null}],
 "injectors": [{"block": 1, "mode": "water_rate", "target": 150.0}],
 "provenance": []
}
```

- `grid`: `dx` has one value per column (nx), `dy` per row (ny), `dz` per layer (nz). `top_depth` is the depth of
  the top of layer 1: one number, or one per column of blocks (`nx*ny` numbers) for a dipping structure.
- `initial` is either `{"type": "equilibrium", "datum_pressure": ..., "woc_depth": ...}` (capillary-gravity
  equilibrium; needs a capillary pressure table) or `{"type": "uniform", "pressure": ..., "sw": ...}`.
- Producer `mode`: `oil_rate`, `liquid_rate`, `bhp`. Injector `mode`: `water_rate`, `bhp`. `block` is one block
  number or a list of block numbers (a well completed in several layers).
- Required: `grid`, `kx`, `ky`, `kz`, `porosity`, `rock_compressibility`, `pvt`, `rock_fluid` (without
  `pc_multiplier`), `gamma_o`, `gamma_w`, `swc`, `sor`, `initial`, `dt`, `t_end`; and in a well `block`, `mode`,
  `target`. Everything else has the default of the Python class (see the docstrings of `Producer` and `Injector`).
- When a model is written, the capillary-pressure column is the table the simulator uses (already multiplied by
  `pc_multiplier`) and `pc_multiplier` is written as 1.

## Schedule

A list of well-control events (see `kusim.schedule`). `mode` and `target` are each optional but at least one must be
given; times must be multiples of `dt`.

```json
[{"time": 200, "well": "P1", "target": 6000.0},
 {"time": 400, "well": "I1", "mode": "bhp", "target": 4800.0}]
```

## Observations

A list of records for history matching: `time`, `vector` (a summary vector such as `WBHP:P1`), `value`, `sigma`
(standard deviation of the error) and optionally `weight`.

## Results

`results.json` holds `grid`, `completions`, `stop_reason`, the tables `wells` and `perforations` (a list of records,
one per well per time step; columns as in `Results`), and the maps `snapshot_times`, `pressure`, `sw`
(each map is a flat list of `n_cells` values). How many maps are stored is set by `fields`:
`"none"`, `"last"` (default) or `"all"` (large for big grids).

`summary.json` (and the standard output of `kusim run`) is a short account for a program that does not want the
tables: for each well the values at the last step, plus `diagnostics` and `healthy`. `healthy` is `false` when the
pressure left the PVT table, saturations were clipped, a well flowed backwards, or a stop condition ended the run;
see `Simulator.diagnostics` before trusting a run.

## Python

```python
import kusim
model = kusim.load_model("model.json")             # or ReservoirModel.from_dict(d)
results = kusim.Simulator(model).run()
results.to_json("results.json", fields="last")     # or results.to_dict()
model.to_json("copy.json")
```

## Command line

```
kusim template -o model.json                    an example model to start from
kusim convert INPUT.DAT -o model.json           a legacy input file as JSON
kusim validate model.json [--schedule s.json]   check without running
kusim run model.json --out results/ [--schedule s.json] [--fields none|last|all] [--csv]
                     [--solver sparse|gauss] [--upstream potential|pressure] [--save-every N] [--strict]
kusim edit model.json --set PATH=VALUE [--set ...] [--patch p.json] (-o new.json | --in-place)
kusim diff a.json b.json                        what differs between two models, by path
kusim schema model|schedule|observations|results    JSON Schema (draft 2020-12)
kusim version
```

`python -m kusim` is the same. Every command prints **one JSON object** on standard output with `"ok": true`, or
`"ok": false` and an `"error"` message. Exit codes: 0 success; 2 bad input (missing or invalid files, an unknown
well in a schedule, a bad option); 1 only with `run --strict`, when the run finished but `healthy` is false.

### Edit and diff

`kusim edit` changes a model without rewriting it. A path is made of dictionary keys and list positions **counted
from 0** (a position in the list, not a block number): `kx`, `dt`, `initial.sw`, `pvt.bo.2`, `producers.0.target`,
`injectors.0.mode`. A value is read as JSON when it is JSON (`150`, `null`, `[1, 2]`, `"bhp"`) and as text otherwise.
`--patch file.json` merges an object into the model (dictionaries key by key, lists replaced). Nothing is written
unless the edited model is valid: it is rebuilt with the same strict reading as a file, so an unknown key or a bad
value is an error. The answer lists `changes` (`path`, `old`, `new`; long arrays are summarised as `n`, `min`, `max`)
and describes the new model. A uniform array (`"kx": 100`) is stored as one number, so `kx.3=500` is rejected with
an explanation: set `kx` to a full list first.

`kusim diff` lists every difference by path (`provenance` is ignored). A long numeric array that changed is reported
once with `n_changed`, `of` and `max_abs_change`.

### Errors an agent can act on

A failed command prints `"ok": false`, `"error"` (a sentence) and, when the problem can be located, machine-readable
fields:

```json
{"ok": false, "error": "...", "code": "unknown_key", "path": "producers.0.min_bph",
 "allowed": ["block", "min_bhp", "..."], "did_you_mean": {"min_bph": "min_bhp"}}
```

`code` is one of `unknown_key`, `missing_key` (`missing` lists them all), `not_an_object`, `bad_value`, `bad_path`,
`bad_argument`, `invalid_json` (with `line`, `column`), `unsupported_version`, `unsupported_units` or
`invalid_value` (the model rejected the data and there is no single location, for example a block outside the grid).
`path` uses the paths of `kusim edit`; for a schedule or observation list it starts with the item (`[1].rate`).

### Schemas

`kusim schema model` (also `schedule`, `observations`, `results`) prints a JSON Schema. It covers keys, types and
allowed values, and it is kept in step with the readers by a test. Rules that involve several fields (array lengths,
block numbers inside the grid, schedule times that are multiples of `dt`) are checked by `kusim validate`.

## Limits

- The format covers the model, schedules and observations. Restart states (`SimState`) and parameter sets are not
  serialised yet.
- Only the full model is read: there is no partial or "patch" file. To change a model, edit the JSON or use
  `ReservoirModel.with_changes` in Python.
- Only oilfield units; the format has no unit conversion.
