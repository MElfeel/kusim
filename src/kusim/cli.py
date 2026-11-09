"""Command line: run kusim from a shell, a script in any language, or an agent, with JSON files in and out.

    kusim template -o model.json                    write an example model (a small layered waterflood)
    kusim convert INPUT.DAT -o model.json           turn a legacy input file into a JSON model
    kusim validate model.json                       check a model without running it
    kusim run model.json --out results/             run it; writes results/results.json and prints a summary
    kusim run model.json --schedule schedule.json   ... with well-control changes in time
    kusim run model.json --until 200 --state-out s.json    run to day 200 and keep the state;
    kusim run model.json --restart s.json --schedule x.json   ... continue from it (changing wells on the way)
    kusim misfit model.json --observations obs.json   run it and compare with data: objective, rms, worst residuals
    kusim gradient model.json --observations obs.json --parameters p.json   adjoint gradient of the misfit
    kusim ensemble model.json --method esmda|enkf --observations o.json --parameters p.json --prior prior.json
    kusim serve --workspace ./work                    the same commands over HTTP (pip install kusim[api]; see kusim.api)
    kusim edit model.json --set kx=150 -o new.json  change values by path (or --patch file, --in-place)
    kusim diff a.json b.json                        what differs between two models, by path
    kusim tables FILE                               the tables and columns of a model, results, observations or misfit file
    kusim plot-template [--name N -o spec.json]     worked plot descriptions to start from
    kusim plot spec.json -o fig.png                 a figure from a JSON description (queries + a Plotly figure; docs/plotting.md)
    kusim describe model.json                       what is in a model: grid, properties, wells with their control and units
    kusim refine model.json --factor 2 -o fine.json   cut every block into smaller ones (wells and properties follow)
    kusim compare runs/a runs/b                     two finished runs, well by well (values, change, percent, notes)
    kusim schema model                              the JSON Schema of model, schedule, observations or results
    kusim version

Every command prints **one JSON object** on standard output (``"ok": true`` or ``"ok": false`` with an
``"error"`` message), so that a caller can parse the answer without scraping text.  The exit code is 0 on
success, 2 for bad input (unreadable or invalid files, unknown options) and, with ``run --strict``, 1 when the run
finished but its diagnostics show signs of trouble (``"healthy": false`` in the summary).  ``python -m kusim`` does the same as ``kusim``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .checks import DEFAULT_MAX_CELLS, DEFAULT_MAX_OUTPUT_MB, DEFAULT_MAX_SECONDS, check_model
from .schema import SCHEMAS, get_schema
from .solver import METHODS
from .spec import (FIELD_MODES, SpecError, _clean, load_model, load_observations, load_schedule, load_state, save_schedule, _write_json, model_to_dict, run_summary, save_model,
                   save_results, save_state)

EXIT_BAD_INPUT = 2
EXIT_TROUBLE = 1


class _InputError(Exception):
    """Something is wrong with what the caller gave us (``details`` goes into the JSON answer: code, path, ...)."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def _print(data):
    print(json.dumps(_clean(data), indent=1))


def _well_names(model):
    return ([f"P{n}" for n in range(1, len(model.producers) + 1)]
            + [f"I{n}" for n in range(1, len(model.injectors) + 1)])


def _describe(model):
    g = model.grid
    return {
        "n_cells": g.n_cells, "shape_nx_ny_nz": [g.nx, g.ny, g.nz], "dt": model.dt, "t_end": model.t_end,
        "n_steps": model.n_steps,
        "producers": [{"name": f"P{n}", "blocks": w.blocks, "mode": w.mode, "target": w.target,
                       "target_unit": _target_unit(w.mode)} for n, w in enumerate(model.producers, start=1)],
        "injectors": [{"name": f"I{n}", "blocks": w.blocks, "mode": w.mode, "target": w.target,
                       "target_unit": _target_unit(w.mode)} for n, w in enumerate(model.injectors, start=1)],
    }


def _target_unit(mode):
    return "psi" if mode == "bhp" else "STB/day"


def _control_warnings(model, changes):
    """Warnings about an edit that acts on nothing: no value changed, or a setting that is valid but inert (see checks.inert_settings)."""
    from .checks import inert_settings
    from .edit import _steps

    warnings = []
    if changes and all(c["old"] == c["new"] for c in changes):
        warnings.append("no value changed: every edited value was already what you set, so the new file is the same as the old one")
    touched = {".".join(_steps(c["path"])) for c in changes}
    warnings += [f["message"] for f in inert_settings(model) if f["path"] in touched]
    return warnings


def _details(error):
    """The machine-readable part of a SpecError (code, path, hints) without the message."""
    return {k: v for k, v in error.as_dict().items() if k != "error"}


def _add_guard_arguments(p, output=False):
    p.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS,
                   help=f"refuse models with more blocks (default {DEFAULT_MAX_CELLS}; 0 = no limit)")
    p.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                   help=f"refuse runs estimated to take longer (default {DEFAULT_MAX_SECONDS:g}; 0 = no limit)")
    if output:
        p.add_argument("--max-output-mb", type=float, default=DEFAULT_MAX_OUTPUT_MB,
                       help=f"refuse results.json estimated above this (default {DEFAULT_MAX_OUTPUT_MB:g}; 0 = no limit)")


def _guard(model, args, **kwargs):
    """Check before running (see kusim.checks). Raises _InputError on an error finding; returns what an answer shows:
    the warnings, the stability number and the runtime estimate."""
    check = check_model(model, max_cells=args.max_cells, max_seconds=args.max_seconds,
                        max_output_mb=getattr(args, "max_output_mb", 0.0), upstream=getattr(args, "upstream", "potential"),
                        linear_solver=getattr(args, "solver", "sparse"), **kwargs)
    errors = [f for f in check["findings"] if f["severity"] == "error"]
    if errors:
        first = errors[0]
        raise _InputError(first["message"], code=first["code"], findings=check["findings"], estimate=check["estimate"])
    return {"findings": check["findings"], "stability": check["stability"], "estimate": check["estimate"]}


def _steps_to(model, last_time, start=0.0):
    return max(int(round((last_time - start) / model.dt)), 0)


def _load(path, loader, what):
    try:
        return loader(path)
    except FileNotFoundError:
        folder = Path(path).resolve().parent
        nearby = sorted(p.name for p in folder.iterdir() if p.is_file())[:20] if folder.is_dir() else []
        raise _InputError(f"{what} file not found: {path}", code="file_not_found", file=str(path), files_nearby=nearby,
                          hint="check the name against files_nearby; if this file was optional (an option such as --schedule), "
                               "leave the option out instead of naming a file that does not exist") from None
    except SpecError as error:
        raise _InputError(f"{what} {path} is not valid: {error}", **_details(error)) from None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise _InputError(f"{what} {path} is not valid: {error}", code="invalid_value") from None


def _check_schedule(model, schedule, path):
    """The schedule must name wells the model has and fit the time step (raises _InputError)."""
    from .schedule import plan_advance

    names = _well_names(model)
    unknown = sorted({e.well for e in schedule.events} - set(names))
    if unknown:
        raise _InputError(f"schedule {path} names wells that the model does not have: {unknown} "
                          f"(the model has {names})")
    try:
        plan_advance(0.0, model.t_end, model.dt, model.t_end, schedule)
    except ValueError as error:
        raise _InputError(f"schedule {path} does not fit the model: {error}") from None


# ---------------------------------------------------------------------- commands
def _cmd_version(args):
    _print({"ok": True, "kusim": __version__})


def _cmd_template(args):
    from .cases import layered_waterflood

    model = layered_waterflood(nx=args.nx, ny=args.ny, nz=args.nz, days=args.days)
    if args.output:
        save_model(model, args.output)
        _print({"ok": True, "written": str(args.output), **_describe(model)})
    else:
        _print(model_to_dict(model))


def _cmd_convert(args):
    from .io import read_legacy_input

    model = _load(args.input, read_legacy_input, "legacy input")
    save_model(model, args.output)
    _print({"ok": True, "written": str(args.output), **_describe(model)})


def _cmd_import_deck(args):
    from .deck import read_deck

    pressures = [float(x) for x in args.pvt_pressures.split(",")] if args.pvt_pressures else None
    imported = _load(args.deck, lambda path: read_deck(path, dt=args.dt, skip_unknown=args.skip_unknown,
                                                       pvt_pressures=pressures), "deck")
    model, warnings = imported.model, list(imported.warnings)
    save_model(model, args.output)
    info = {"ok": True, "written": str(args.output), **_describe(model), "wells": imported.wells,
            "n_events": len(imported.schedule), "mapping": imported.mapping, "ignored": imported.ignored}
    if args.schedule_out:
        save_schedule(imported.schedule, args.schedule_out)
        info["schedule_written"] = str(args.schedule_out)
    elif len(imported.schedule):
        warnings.append({"code": "schedule_not_written", "where": args.deck, "message":
                         f"the deck changes well controls {len(imported.schedule)} times: give --schedule-out to write them"})
    if args.wells_out:
        _write_json(imported.wells, args.wells_out)
        info["wells_written"] = str(args.wells_out)
    check = check_model(model, probe_steps=0 if args.no_probe else 3)
    info["runnable"] = check["runnable"]
    info["checks"] = {k: check[k] for k in ("findings", "stability", "estimate", "output_mb")}
    info["warnings"] = warnings
    _print(info)


def _pairs(items, what):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise _InputError(f"{what} {item!r} must look like NAME=VALUE", code="bad_value")
        out[key.strip()] = value.strip()
    return out


def _cmd_import_history(args):
    from .excel import read_history
    from .spec import _read_json, observations_to_list, schedule_to_list

    model = _load(args.model, load_model, "model") if args.model else None
    wells = _pairs(args.well_map, "--well-map")
    if args.wells_file:
        wells = {**_load(args.wells_file, _read_json, "wells"), **wells}
    sigma = _pairs(args.sigma, "--sigma")
    try:
        sigma = {k: float(v) if not v.endswith("%") else v for k, v in sigma.items()}
    except ValueError:
        raise _InputError("--sigma values are numbers or percentages (WBHP=25, WOPR=5%)", code="bad_value") from None
    imported = _load(args.input, lambda path: read_history(
        path, sheet=args.sheet, layout=args.layout, header_row=args.header_row, start_date=args.start_date,
        dayfirst=args.dayfirst, model=model, dt=args.dt, every=args.every, align=args.align, wells=wells,
        vectors=args.vectors.split(",") if args.vectors else None, pressure_unit=args.pressure_unit,
        rate_unit=args.rate_unit, water_cut_unit=args.water_cut_unit, sigma=sigma, controls=bool(args.controls_out),
        producer_control=args.producer_control, skip_wells=args.skip_well), "history")
    _write_json(observations_to_list(imported.observations), args.output)
    info = {"ok": True, "written": str(args.output), **imported.report}
    if args.controls_out:
        _write_json(schedule_to_list(imported.schedule), args.controls_out)
        info["controls_written"] = str(args.controls_out)
    _print(info)


def _cmd_history_template(args):
    from .excel import write_history_template

    try:
        write_history_template(args.output)
    except ImportError:
        raise _InputError("writing Excel needs openpyxl: pip install 'kusim[excel]'", code="missing_dependency") from None
    _print({"ok": True, "written": str(args.output)})


def _cmd_validate(args):
    model = _load(args.model, load_model, "model")
    info = {"ok": True, **_describe(model)}
    if args.schedule:
        schedule = _load(args.schedule, load_schedule, "schedule")
        _check_schedule(model, schedule, args.schedule)
        info["n_events"] = len(schedule)
    if args.restart:
        from .simulator import Simulator

        info["restart_time"] = _restart(Simulator(model), args.restart)
    check = check_model(model, probe_steps=0 if args.no_probe else 3, max_cells=args.max_cells,
                        max_seconds=args.max_seconds, max_output_mb=args.max_output_mb, fields=args.fields)
    info["runnable"] = check["runnable"]
    info["checks"] = {k: check[k] for k in ("findings", "stability", "estimate", "output_mb")}
    _print(info)


def _cmd_edit(args):
    from .edit import apply_edits, parse_value
    from .spec import _read_json

    data = _load(args.model, _read_json, "model")
    if isinstance(data, dict) and "figure" in data and "sources" in data:
        raise _InputError("this file is a plot description, not a model: `edit` is for models. Write the plot description again with "
                          "`plot-template --name NAME -o FILE --source ROLE=FILE [--quantity Q] [--well W] [--title T]`", code="not_a_model")
    assignments = []
    for text in args.assignments:
        path, sep, value = text.partition("=")
        if not sep:
            raise _InputError(f"--set {text!r} must look like PATH=VALUE", code="bad_argument")
        assignments.append((path.strip(), parse_value(value)))
    patch = _load(args.patch, _read_json, "patch") if args.patch else None
    if not assignments and patch is None:
        raise _InputError("nothing to do: give at least one --set or a --patch", code="bad_argument")
    try:
        model, _, changes = apply_edits(data, assignments, patch)
    except SpecError as error:
        raise _InputError(f"the edit is not valid: {error}", **_details(error)) from None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise _InputError(f"the edited model is not valid: {error}", code="invalid_value") from None
    out = args.model if args.in_place else args.output
    save_model(model, out)
    answer = {"ok": True, "written": str(out), "changes": changes, **_describe(model)}
    warnings = _control_warnings(model, changes)
    if warnings:
        answer["warnings"] = warnings
    _print(answer)


def _cmd_diff(args):
    from .edit import diff_models

    a, b = _load(args.a, load_model, "model"), _load(args.b, load_model, "model")
    changes = diff_models(a, b)
    _print({"ok": True, "identical": not changes, "n_differences": len(changes), "differences": changes})


def _cmd_describe(args):
    import numpy as np

    model = _load(args.model, load_model, "model")
    g = model.grid
    info = {"ok": True, **_describe(model)}
    info["size_ft"] = {"x": float(g.dx[: g.nx].sum()), "y": float(g.dy[: g.n_layer : g.nx].sum()), "z": float(g.dz[:: g.n_layer].sum())}
    info["block_size_ft"] = {"dx": sorted({float(v) for v in g.dx[: g.nx]}), "dy": sorted({float(v) for v in g.dy[: g.n_layer : g.nx]}),
                             "dz": sorted({float(v) for v in g.dz[:: g.n_layer]})}
    info["properties"] = {name: {"min": float(np.min(v)), "max": float(np.max(v))}
                          for name, v in (("kx_md", model.kx), ("ky_md", model.ky), ("kz_md", model.kz), ("porosity", model.porosity))}
    info["initial"] = model_to_dict(model)["initial"]
    wells = model.producers + model.injectors
    for well in [*info["producers"], *info["injectors"]]:
        w = wells[_well_names(model).index(well["name"]) if well["name"].startswith("P") else len(model.producers) + int(well["name"][1:]) - 1]
        well["ijk"] = [list(g.block_ijk(b)) for b in w.blocks]
        for key in ("min_bhp", "min_oil_rate"):
            if getattr(w, key, None) is not None:
                well[key] = getattr(w, key)
    info["how_to_change_a_well"] = ("kusim edit MODEL --set producers.0.target=VALUE -o NEW (or P1.target=VALUE). 'target' is the "
                                    "bottom-hole pressure in psi when mode is 'bhp', and the rate in STB/day for a rate mode. "
                                    "'min_bhp' is only a lower limit for a producer that is on a rate mode.")
    _print(info)


def _cmd_refine(args):
    from .refine import refine_model

    model = _load(args.model, load_model, "model")
    fx = args.fx or args.factor
    fy, fz = args.fy or args.factor, args.fz or args.factor
    n_new = model.grid.n_cells * fx * fy * fz
    if args.max_cells and n_new > args.max_cells:
        raise _InputError(f"the refined model would have {n_new} blocks, over the limit of {args.max_cells}; "
                          "refine by less or raise --max-cells if the user wants that", code="too_large", n_cells=n_new)
    dt = args.dt if args.dt in ("auto", "keep") else _number(args.dt, "--dt")
    try:
        new, info = refine_model(model, fx, fy, fz, dt=dt)
    except ValueError as error:
        raise _InputError(str(error), code="bad_argument") from None
    save_model(new, args.output)
    _print({"ok": True, "written": str(args.output), **info, **_describe(new)})


def _number(text, what):
    try:
        return float(text)
    except ValueError:
        raise _InputError(f"{what} must be 'auto', 'keep' or a number of days, got {text!r}", code="bad_argument") from None


def _cmd_compare(args):
    from .compare import compare_summaries, load_summary

    summaries = []
    for path in (args.a, args.b):
        try:
            summaries.append(load_summary(path))
        except FileNotFoundError as error:
            raise _InputError(f"run summary not found: {error}", code="file_not_found", file=str(path),
                              hint="give the folder of a run written by 'kusim run MODEL --out FOLDER' (it holds summary.json)") from None
        except ValueError as error:
            raise _InputError(str(error), code="invalid_value") from None
    _print({"ok": True, "a": str(args.a), "b": str(args.b), **compare_summaries(*summaries)})


def _cmd_tables(args):
    from .queries import source_from_data
    from .spec import _read_json

    data = _load(args.file, _read_json, "file")
    try:
        source = source_from_data(data, args.file)
        tables = source.describe()
    except SpecError as error:
        raise _InputError(str(error), **_details(error)) from None
    except (ValueError, KeyError, TypeError) as error:
        raise _InputError(f"{args.file} could not be read as a {detect_name(data)}: {error}", code="invalid_value") from None
    _print({"ok": True, "file": str(args.file), "kind": source.kind, "tables": tables,
            "usage": 'in a plot description: "sources": {"run": "' + str(args.file) + '"}, then a query {"from": "run.<table>", ...} or '
                     'a ref {"$ref": "run.<table>.<column>"}; see docs/plotting.md'})


def detect_name(data):
    from .queries import detect_kind

    return detect_kind(data) or "kusim file"


def _cmd_plot_template(args):
    from .figure import EXAMPLE_INFO, customise, example_spec

    if not args.name:
        _print({"ok": True, "examples": [{"name": n, "description": d, "needs": needs} for n, (d, needs) in EXAMPLE_INFO.items()],
                "usage": "kusim plot-template --name NAME -o spec.json --source ROLE=FILE ... [--quantity WWCT] [--well P1] [--title T] writes one "
                         "with your files (the roles are listed in the answer); then kusim plot spec.json -o fig.png. Do NOT use `kusim edit` on it "
                         "(that is for models). Use kusim tables FILE to see the columns of each file. Details: docs/plotting.md"})
        return
    if not args.output:
        raise _InputError("give -o FILE to write the example", code="bad_argument")
    sources = {}
    for item in args.source or []:
        role, sep, file = item.partition("=")
        if not sep:
            raise _InputError(f"--source {item!r} must look like ROLE=FILE", code="bad_argument")
        sources[role.strip()] = file.strip()
    try:
        text = customise(example_spec(args.name), sources, args.quantity, args.well, args.title)
    except SpecError as error:
        raise _InputError(str(error), **_details(error)) from None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf8")
    description, needs = EXAMPLE_INFO[args.name]
    roles = json.loads(text)["sources"]
    _print({"ok": True, "written": str(args.output), "description": description, "needs": needs, "sources": roles,
            "next": f"a 'run' role needs the results.json of a RUN folder (kusim run MODEL --out DIR gives DIR/results.json), not a model file. "
                    f"Change a role with --source ROLE=FILE (write the template again). Then: kusim plot {args.output} -o fig.png"})


def _cmd_plot(args):
    from .figure import prepare, render
    from .spec import _read_json

    spec = _load(args.spec, _read_json, "plot description")
    try:
        bound, _, info = prepare(spec)
        written = render(bound, args.output, args.width, args.height, args.scale)
    except SpecError as error:
        raise _InputError(f"the plot description is not valid: {error}", **_details(error)) from None
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise _InputError(f"the plot description is not valid: {error}", code="invalid_value") from None
    _print({"ok": True, **written, **info})


def _cmd_schema(args):
    _print(get_schema(args.name))


def _restart(sim, path):
    """Continue ``sim`` from a saved state file; returns the start time (raises _InputError if it does not fit)."""
    state = _load(path, load_state, "state")
    try:
        sim.set_state(state)
    except ValueError as error:
        raise _InputError(f"state {path} does not fit the model: {error}", code="state_mismatch") from None
    return sim.time


def _check_observations(model, observations, start, path):
    """The observations must name vectors the model has and times the run reaches (raises _InputError)."""
    import difflib

    from .history import vector_names

    known = vector_names(model)
    for i, r in enumerate(observations.table.itertuples()):
        if r.vector not in known:
            close = difflib.get_close_matches(r.vector, known, n=1)
            raise _InputError(f"observations {path}: the model has no vector {r.vector!r}; it has {known}",
                              code="unknown_vector", vector=r.vector, allowed=known,
                              did_you_mean={r.vector: close[0]} if close else {})
        steps = r.time / model.dt
        if abs(steps - round(steps)) > 1e-6 or not (start < r.time <= model.t_end + 1e-9):
            raise _InputError(f"observations {path}: time {r.time} must be a multiple of dt = {model.dt} after the "
                              f"start (day {start}) and at most t_end = {model.t_end}", code="bad_time", time=r.time)


def _cmd_misfit(args):
    from .history import misfit, objective, rms, run_to
    from .simulator import Simulator, diagnostics_healthy
    from .spec import load_observations

    model = _load(args.model, load_model, "model")
    observations = _load(args.observations, load_observations, "observations")
    schedule = _load(args.schedule, load_schedule, "schedule") if args.schedule else None
    if schedule is not None:
        _check_schedule(model, schedule, args.schedule)
    state, start = None, 0.0
    if args.restart:
        state = _load(args.restart, load_state, "state")
        try:
            probe = Simulator(model)
            probe.set_state(state)
            start = probe.time
        except ValueError as error:
            raise _InputError(f"state {args.restart} does not fit the model: {error}", code="state_mismatch") from None
    _check_observations(model, observations, start, args.observations)
    checks = _guard(model, args, fields="none", n_steps=_steps_to(model, max(observations.times), start))
    try:
        record = run_to(model, observations.times, schedule, observations.vectors, state, upstream=args.upstream)
    except ValueError as error:
        raise _InputError(f"cannot run the model: {error}", code="invalid_value") from None
    reached = set(record.table.TIME.round(9)) if len(record.table) else set()
    have = observations.table[observations.table.time.round(9).isin(reached)]
    missing = observations.table[~observations.table.time.round(9).isin(reached)]
    table = misfit(type(observations)(have), record.table) if len(have) else None
    healthy = bool(not len(missing) and diagnostics_healthy(record.diagnostics))
    out = {"ok": True, "n_observations": len(observations.table), "n_compared": 0 if table is None else len(table),
           "n_missing": len(missing), "final_time": record.state.time, "diagnostics": record.diagnostics,
           "healthy": healthy, "checks": checks}
    if len(missing):
        out["missing_times"] = sorted(set(float(t) for t in missing.time))
        out["note"] = "the run ended before these observation times (see stop_reason in diagnostics)"
    if table is not None:
        by_vector = {}
        for name, group in table.groupby("vector"):
            by_vector[name] = {"n": len(group), "rms": float((group.normalised ** 2).mean() ** 0.5),
                               "mean_residual": float(group.residual.mean()),
                               "worst_time": float(group.loc[group.normalised.abs().idxmax(), "time"])}
        worst = table.reindex(table.normalised.abs().sort_values(ascending=False).index).head(10)
        out.update(objective=objective(table), rms=rms(table), by_vector=by_vector,
                   worst=worst.to_dict("records"))
        if len(table) <= 50:
            out["table"] = table.to_dict("records")
        if args.out:
            _write_json(table.to_dict("records"), args.out)
            out["written"] = str(args.out)
    _print(out)
    return EXIT_TROUBLE if (args.strict and not healthy) else 0


def _gradient_problem(args, model, observations, schedule, state, start):
    from .gradients import HistoryProblem
    from .parameters import ScalarParameter
    from .spec import controls_from_dict, parameters_from_dict, _read_json

    params = _load(args.parameters, lambda p: parameters_from_dict(_read_json(p)), "parameters") if args.parameters else None
    controls = initial = None
    if args.controls:
        controls, initial = _load(args.controls, lambda p: controls_from_dict(_read_json(p)), "controls")
        names = _well_names(model)
        for i, p in enumerate(controls.parameters):
            if p.well not in names:
                raise _InputError(f"controls {args.controls}: the model has no well {p.well!r} (it has {names})",
                                  code="unknown_well", path=f"controls.{i}.well", allowed=names)
            for t in p.times:
                steps = t / model.dt
                if abs(steps - round(steps)) > 1e-6 or not (0 <= t < model.t_end):
                    raise _InputError(f"controls {args.controls}: period start {t} of {p.well} must be a multiple of "
                                      f"dt = {model.dt} before t_end", code="bad_time", path=f"controls.{i}.times")
    if params is None and controls is None:
        raise _InputError("nothing to differentiate with respect to: give --parameters and/or --controls",
                          code="bad_argument")
    if args.method == "adjoint" and params is not None and any(isinstance(p, ScalarParameter) for p in params.parameters):
        raise _InputError("the adjoint has no gradient for scalar parameters; use --method fd for those "
                          "(slower: one or two runs per component)", code="no_gradient")
    try:
        problem = HistoryProblem(model, observations, params=params, controls=controls, base_schedule=schedule,
                                 state=state, initial_controls=initial, upstream=args.upstream)
    except (ValueError, KeyError, AttributeError, TypeError) as error:
        raise _InputError(f"cannot set up the problem: {error}", code="invalid_value") from None
    return problem, params, controls


def _cmd_gradient(args):
    import numpy as np

    from .simulator import Simulator
    from .spec import _read_json

    model = _load(args.model, load_model, "model")
    observations = _load(args.observations, load_observations, "observations")
    schedule = _load(args.schedule, load_schedule, "schedule") if args.schedule else None
    if schedule is not None:
        _check_schedule(model, schedule, args.schedule)
    state, start = None, 0.0
    if args.restart:
        state = _load(args.restart, load_state, "state")
        probe = Simulator(model)
        try:
            probe.set_state(state)
        except ValueError as error:
            raise _InputError(f"state {args.restart} does not fit the model: {error}", code="state_mismatch") from None
        start = probe.time
    _check_observations(model, observations, start, args.observations)
    problem, params, controls = _gradient_problem(args, model, observations, schedule, state, start)
    n_variables = problem.n_params + problem.n_controls
    checks = _guard(model, args, fields="none", n_steps=_steps_to(model, max(observations.times), start),
                    repeats=3 if args.method == "adjoint" else 2 * n_variables + 1)
    try:
        x = problem.vector0() if not args.x else np.asarray(_load(args.x, _read_json, "vector"), dtype=float)
        if args.x:
            problem.split(x)                                   # checks the length
        if args.method == "adjoint":
            value, grad, result = problem.value_and_gradient(x)
            diagnostics = result.diagnostics
        else:
            value, grad, diagnostics = problem.value(x), problem.fd_gradient(x), None
    except (ValueError, KeyError, NotImplementedError, TypeError) as error:
        raise _InputError(f"cannot compute the gradient: {error}", code="invalid_value") from None

    layout = {}
    if params is not None:
        layout.update({name: ("parameter", sl) for name, sl in params.layout(model).items()})
    if controls is not None:
        for well, sl in controls._slices().items():
            layout[f"{well}_targets"] = ("control", slice(sl.start + problem.n_params, sl.stop + problem.n_params))
    variables = {}
    for name, (kind, sl) in layout.items():
        g = grad[sl]
        entry = {"kind": kind, "n": int(g.size), "norm": float(np.linalg.norm(g)), "max_abs": float(np.abs(g).max()),
                 "index_of_max": int(np.abs(g).argmax()), "mean": float(g.mean())}
        if g.size <= 20:
            entry["gradient"] = g.tolist()
            entry["x"] = x[sl].tolist()
        variables[name] = entry
    out = {"ok": True, "objective": value, "method": args.method, "n_variables": int(x.size),
           "x_from": args.x or "the model (and the initial controls)", "variables": variables,
           "diagnostics": diagnostics, "checks": checks}
    if args.out:
        _write_json({"names": {n: [sl.start, sl.stop] for n, (_, sl) in layout.items()}, "x": x.tolist(),
                     "gradient": grad.tolist(), "objective": value}, args.out)
        out["written"] = str(args.out)
    if args.check:
        if args.method != "adjoint":
            raise _InputError("--check compares the adjoint with finite differences: use --method adjoint",
                              code="bad_argument")
        table = problem.check(x, n=args.check)
        stable = table[table.stable]
        out["check"] = {"n_checked": len(table), "n_stable": len(stable),
                        "max_relative_error": float(stable.relative.max()) if len(stable) else None,
                        "components": table.to_dict("records")}
    _print(out)
    return 0


def _cmd_ensemble(args):
    import numpy as np

    from .ensemble import check_prior, run_enkf, run_esmda
    from .spec import _read_json, load_prior, parameters_from_dict

    model = _load(args.model, load_model, "model")
    observations = _load(args.observations, load_observations, "observations")
    params = _load(args.parameters, lambda p: parameters_from_dict(_read_json(p)), "parameters")
    prior = _load(args.prior, load_prior, "prior")
    schedule = _load(args.schedule, load_schedule, "schedule") if args.schedule else None
    if schedule is not None:
        _check_schedule(model, schedule, args.schedule)
    _check_observations(model, observations, 0.0, args.observations)
    if args.members < 2:
        raise _InputError("--members must be at least 2", code="bad_argument")
    alphas = None
    if args.alphas:
        try:
            alphas = [float(a) for a in args.alphas.split(",")]
        except ValueError:
            raise _InputError(f"--alphas must be numbers separated by commas, got {args.alphas!r}", code="bad_argument") from None
    if args.method == "enkf" and (alphas or args.passes != 4):
        raise _InputError("--passes and --alphas belong to --method esmda", code="bad_argument")
    if args.method == "esmda" and args.localise is not None:
        raise _InputError("--localise belongs to --method enkf (localisation made the stacked ES-MDA update overshoot: "
                          "see docs/history-matching.md)", code="bad_argument")
    passes = args.passes if args.method == "esmda" else 2
    batches = -(-args.members // max(args.workers, 1))
    checks = _guard(model, args, fields="none", n_steps=_steps_to(model, max(observations.times)),
                    repeats=batches * (passes + 1))
    try:
        check_prior(prior, params)
    except ValueError as error:
        raise _InputError(f"prior {args.prior} does not fit the parameters: {error}", code="prior_mismatch",
                          allowed=[p.name for p in params.parameters]) from None
    try:
        if args.method == "esmda":
            result = run_esmda(model, params, prior, observations, args.members, args.seed, alphas, args.passes,
                               schedule, args.workers)
        else:
            result = run_enkf(model, params, prior, observations, args.members, args.seed, schedule, args.workers,
                              args.localise)
    except (ValueError, KeyError, AttributeError, TypeError, np.linalg.LinAlgError) as error:
        raise _InputError(f"the ensemble run failed: {error}", code="invalid_value") from None

    metrics = result["metrics"]
    warnings = []
    if args.members < 10:
        warnings.append("fewer than 10 members: the covariances are estimated from very few samples")
    if metrics["posterior_ok"] < metrics["members"]:
        warnings.append(f"{metrics['members'] - metrics['posterior_ok']} posterior members show signs of trouble "
                        "(pressure outside the PVT table or a well stop condition): read their diagnostics before trusting the fit")
    if metrics.get("saturations_clipped"):
        warnings.append("the EnKF update pushed saturations outside their limits; they were clipped")
    if metrics["posterior_member_rms"] > metrics["prior_member_rms"]:
        warnings.append("the data fit got worse: check the prior (too narrow or wrong mean?) and the observation sigmas")
    healthy = metrics["posterior_ok"] == metrics["members"] and metrics["posterior_mean_ok"]
    out = {"ok": True, "method": args.method, "seed": args.seed, "healthy": healthy, "warnings": warnings,
           "metrics": metrics, "checks": checks}
    if args.out:
        folder = Path(args.out)
        save_model(result["posterior_mean_model"], folder / "posterior_mean.json")
        written = {"posterior_mean_model": str(folder / "posterior_mean.json"),
                   "summary": str(folder / "ensemble.json")}
        if args.write_members:
            for j, m in enumerate(result["posterior_models"]):
                save_model(m, folder / "members" / f"{j:04d}.json")
            written["members"] = str(folder / "members")
        if args.write_ensemble:
            layout = {n: [s.start, s.stop] for n, s in params.layout(model).items()}
            _write_json({"names": layout,
                         "prior": params.matrix(result["prior_models"]).T.tolist(),
                         "posterior": params.matrix(result["posterior_models"]).T.tolist()},
                        folder / "ensemble_parameters.json", indent=None)
            written["parameters"] = str(folder / "ensemble_parameters.json")
        out["written"] = written
        (folder / "ensemble.json").write_text(json.dumps(_clean(out), indent=1) + "\n", encoding="utf8")
    _print(out)
    return EXIT_TROUBLE if (args.strict and not healthy) else 0


def _cmd_serve(args):
    try:
        import uvicorn

        from .api import ServerConfig, create_app
    except ImportError as error:
        raise _InputError(f"the HTTP API needs extra packages ({error}): pip install kusim[api]",
                          code="missing_dependency") from None
    token = args.token or os.environ.get("KUSIM_API_TOKEN")
    config = ServerConfig(workspace=Path(args.workspace), token=token, timeout_seconds=args.timeout,
                          max_concurrent=args.max_concurrent, max_workers=args.max_workers, max_cells=args.max_cells,
                          max_seconds=args.max_seconds, max_output_mb=args.max_output_mb)
    app = create_app(config)
    _print({"ok": True, "serving": f"http://{args.host}:{args.port}/v1", "docs": f"http://{args.host}:{args.port}/docs",
            "workspace": str(config.workspace.resolve()), "token_required": bool(token),
            "timeout_seconds": config.timeout_seconds, "max_concurrent": config.max_concurrent})
    sys.stdout.flush()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def model_hash(model):
    """A short fingerprint of what a run simulated (the model without its provenance), so that two runs of the same model can be told."""
    import hashlib

    data = model_to_dict(model)
    data.pop("provenance", None)
    data = json.loads(json.dumps(data), parse_int=float)          # 2500 and 2500.0 are the same number
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf8")).hexdigest()[:12]


def _cmd_run(args):
    from .schedule import advance
    from .simulator import Simulator

    model = _load(args.model, load_model, "model")
    schedule = _load(args.schedule, load_schedule, "schedule") if args.schedule else None
    if schedule is not None:
        _check_schedule(model, schedule, args.schedule)
    sim = Simulator(model, linear_solver=args.solver, save_every=args.save_every, upstream=args.upstream)
    start = 0.0
    if args.restart:
        start = _restart(sim, args.restart)
    until = model.t_end if args.until is None else args.until
    if until < start - 1e-9 or until > model.t_end + 1e-9:
        raise _InputError(f"--until {until} must lie between the start ({start}) and t_end ({model.t_end})",
                          code="bad_argument")
    checks = _guard(model, args, fields=args.fields, save_every=args.save_every, n_steps=_steps_to(model, until, start))
    wall = time.perf_counter()
    if schedule is None and not args.restart and args.until is None:
        sim.run()
    else:
        try:
            advance(sim, until, schedule)
        except ValueError as error:
            raise _InputError(f"cannot advance to day {until}: {error}", code="invalid_value") from None
    results = sim.results()
    summary = run_summary(results, sim.diagnostics())
    summary.update(final_time=sim.time, start_time=start, n_steps=sim.step_number, solver=args.solver, model_hash=model_hash(model),
                   upstream=args.upstream, kusim=__version__, checks=checks,
                   elapsed_seconds=round(time.perf_counter() - wall, 3))
    if args.state_out:
        save_state(sim.get_state(), args.state_out)
        summary["state"] = str(args.state_out)
    if args.out:
        out = Path(args.out)
        save_results(results, out / "results.json", fields=args.fields)
        written = {"results": str(out / "results.json")}
        if args.csv:
            results.to_csv(out)
            written["csv_folder"] = str(out)
        written["summary"] = str(out / "summary.json")
        summary["written"] = written
        (out / "summary.json").write_text(json.dumps(_clean(summary), indent=1) + "\n", encoding="utf8")
    _print({"ok": True, **summary})
    return EXIT_TROUBLE if (args.strict and summary["healthy"] is False) else 0


# ---------------------------------------------------------------------- parser and main
class _Parser(argparse.ArgumentParser):
    def error(self, message):                                    # a usage error is JSON too
        _print({"ok": False, "error": message})
        sys.exit(EXIT_BAD_INPUT)


def build_parser():
    parser = _Parser(prog="kusim", description="KUSIM two-phase reservoir simulator (JSON in, JSON out).")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    p = sub.add_parser("version", help="print the version")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("template", help="write an example model (layered waterflood) to start from")
    p.add_argument("-o", "--output", help="file to write (default: print the model)")
    p.add_argument("--nx", type=int, default=10)
    p.add_argument("--ny", type=int, default=10)
    p.add_argument("--nz", type=int, default=3)
    p.add_argument("--days", type=float, default=1000.0, help="simulated time")
    p.set_defaults(func=_cmd_template)

    p = sub.add_parser("convert", help="convert a legacy INPUT.DAT file to a JSON model")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("import-deck", help="read an ECLIPSE-style deck (.DATA) into a JSON model and schedule")
    p.add_argument("deck")
    p.add_argument("-o", "--output", required=True, help="model file to write")
    p.add_argument("--schedule-out", help="file for the well-control events after day 0")
    p.add_argument("--wells-out", help="file for the map of deck well names to kusim names (P1, I1, ...)")
    p.add_argument("--dt", type=float, help="time step in days (default: the greatest common divisor of the TSTEP/DATES intervals)")
    p.add_argument("--pvt-pressures", help="comma-separated pressures (psi) of the PVT table when the oil is given by PVCDO")
    p.add_argument("--skip-unknown", action="store_true",
                   help="warn about keywords kusim does not support instead of stopping (use with care)")
    p.add_argument("--no-probe", action="store_true", help="skip running the first steps (no stability or time estimate)")
    p.set_defaults(func=_cmd_import_deck)

    p = sub.add_parser("import-history", help="read historical well data from an Excel workbook or CSV into observations "
                                               "(and well controls)")
    p.add_argument("input", help=".xlsx, .xlsm or .csv file")
    p.add_argument("-o", "--output", required=True, help="observations file to write")
    p.add_argument("--controls-out", help="also write the measured rates as a schedule of well controls")
    p.add_argument("--model", help="model file: well names and kinds, dt and end time")
    p.add_argument("--sheet", help="sheet name (default: the first)")
    p.add_argument("--layout", choices=("auto", "by_well", "wide", "long"), default="auto")
    p.add_argument("--header-row", type=int, default=1, help="row of the column headings (default 1)")
    p.add_argument("--start-date", help="date of day 0 (YYYY-MM-DD; default: the first date in the data)")
    p.add_argument("--dayfirst", action="store_true", help="dates like 03/04/2020 are day first")
    p.add_argument("--dt", type=float, help="time step in days (default: the model's)")
    p.add_argument("--every", type=float, help="observation interval in days (a multiple of dt; default dt)")
    p.add_argument("--align", choices=("mean", "last"), default="mean", help="how data inside an interval are combined")
    p.add_argument("--well-map", action="append", metavar="NAME=KUSIM", help="map a well name of the sheet to P1, I1, ... (repeatable)")
    p.add_argument("--wells-file", help="JSON map of well names (the --wells-out of import-deck)")
    p.add_argument("--skip-well", action="append", metavar="NAME",
                   help="leave this well of the sheet out, on purpose (repeatable; it is reported). Never map two wells of the sheet to one kusim well")
    p.add_argument("--vectors", help="comma-separated vectors to keep (WBHP,WOPR,WWPR,WWCT,WWIR; default all)")
    p.add_argument("--pressure-unit", default="psi", help="psi, bar, kPa, MPa, atm, kgf/cm2 (a unit in the header wins)")
    p.add_argument("--rate-unit", default="stb/d", help="stb/d, bbl/d, m3/d (a unit in the header wins)")
    p.add_argument("--water-cut-unit", default="fraction", help="fraction or percent")
    p.add_argument("--sigma", action="append", metavar="VECTOR=VALUE",
                   help="measurement error of a vector: WBHP=25 (psi) or WOPR=5%% of the value (repeatable)")
    p.add_argument("--producer-control", choices=("liquid_rate", "oil_rate"), default="liquid_rate",
                   help="how --controls-out imposes the producers' measured rates")
    p.set_defaults(func=_cmd_import_history)

    p = sub.add_parser("history-template", help="write an example Excel workbook for import-history")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_history_template)

    p = sub.add_parser("validate", help="check a model (and a schedule) without running")
    p.add_argument("model")
    p.add_argument("--schedule", help="JSON list of well-control events")
    p.add_argument("--restart", metavar="STATE.json", help="also check that this state fits the model")
    p.add_argument("--no-probe", action="store_true", help="skip running the first steps (no stability or time estimate)")
    p.add_argument("--fields", choices=FIELD_MODES, default="last", help="maps that a run would store (output size estimate)")
    _add_guard_arguments(p, output=True)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("edit", help="change values of a model by path (producers.0.target=2500) and write it")
    p.add_argument("model")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", dest="assignments",
                   help="one change as PATH=VALUE, for example producers.0.target=2000 or dt=5 (the value is JSON or text); repeatable; list positions count from 0")
    p.add_argument("--patch", help="NAME OF A FILE (not inline JSON) holding an object merged into the model (lists are replaced); for single values use --set")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("-o", "--output", help="file to write")
    target.add_argument("--in-place", action="store_true", help="overwrite the model file")
    p.set_defaults(func=_cmd_edit)

    p = sub.add_parser("diff", help="list what differs between two models, by path")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=_cmd_diff)

    p = sub.add_parser("describe", help="what is in a model: grid, property ranges, wells with control mode and units")
    p.add_argument("model")
    p.set_defaults(func=_cmd_describe)

    p = sub.add_parser("refine", help="cut every block into smaller ones (e.g. --factor 2 doubles the cells in x, y and z); "
                                      "properties and wells follow; the time step is reduced if the probe says it must")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True, help="refined model file to write")
    p.add_argument("--factor", type=int, default=2, help="parts per block in each direction (default 2: 2 x 2 x 2 = 8 times the blocks)")
    p.add_argument("--fx", type=int, help="parts in x (default: --factor)")
    p.add_argument("--fy", type=int, help="parts in y (default: --factor)")
    p.add_argument("--fz", type=int, help="parts in z (default: --factor)")
    p.add_argument("--dt", default="auto", help="auto (default: smaller only if the stability probe asks), keep, or days")
    p.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS, help=f"refuse a refined model with more blocks (default {DEFAULT_MAX_CELLS}; 0 = no limit)")
    p.set_defaults(func=_cmd_refine)

    p = sub.add_parser("compare", help="compare two finished runs (folders written by run --out), well by well")
    p.add_argument("a", help="run folder (or summary.json) of the first run")
    p.add_argument("b", help="run folder (or summary.json) of the second run")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("tables", help="list the tables and columns that a kusim file offers to a plot (model, results.json, summary.json, "
                                      "observations, misfit table)")
    p.add_argument("file", help="a model, a run's results.json or summary.json, an observations file or the table of 'misfit --out'")
    p.set_defaults(func=_cmd_tables)

    p = sub.add_parser("plot-template", help="list the worked plot descriptions (maps, profiles, observed vs simulated, mismatch per well, ...) "
                                              "or write one to start from")
    p.add_argument("--name", help="the example to write (without it the examples are listed)")
    p.add_argument("-o", "--output", help="file to write the example to")
    p.add_argument("--source", action="append", metavar="ROLE=FILE", help="use FILE for a source of the example (repeatable), e.g. run=out/results.json")
    p.add_argument("--quantity", help="the summary vector to draw, e.g. WWCT or WBHP (examples with a quantity)")
    p.add_argument("--well", help="draw only this well (examples with a quantity)")
    p.add_argument("--title", help="the title of the figure")
    p.set_defaults(func=_cmd_plot_template)

    p = sub.add_parser("plot", help="draw a figure from a JSON description (sources, queries and a Plotly figure) to .png, .svg, .pdf, .html or .json")
    p.add_argument("spec", help="the plot description (docs/plotting.md)")
    p.add_argument("-o", "--output", required=True, help="image or page to write")
    p.add_argument("--width", type=int, default=900, help="width in pixels (default 900)")
    p.add_argument("--height", type=int, default=600, help="height in pixels (default 600)")
    p.add_argument("--scale", type=float, default=2.0, help="pixel density of an image (default 2)")
    p.set_defaults(func=_cmd_plot)

    p = sub.add_parser("schema", help="print the JSON Schema of a file type")
    p.add_argument("name", choices=sorted(SCHEMAS))
    p.set_defaults(func=_cmd_schema)

    p = sub.add_parser("misfit", help="run a model and compare it with observed data")
    p.add_argument("model")
    p.add_argument("--observations", required=True, help="JSON list of observations (kusim schema observations)")
    p.add_argument("--schedule", help="JSON list of well-control events")
    p.add_argument("--restart", metavar="STATE.json", help="start from a saved state (observations must come after it)")
    p.add_argument("--upstream", choices=("potential", "pressure"), default="potential")
    p.add_argument("--out", help="file for the full table of residuals (JSON)")
    _add_guard_arguments(p)
    p.add_argument("--strict", action="store_true", help="exit with 1 if the run shows signs of trouble")
    p.set_defaults(func=_cmd_misfit)

    p = sub.add_parser("gradient", help="objective and gradient of the misfit (adjoint) with respect to parameters/controls")
    p.add_argument("model")
    p.add_argument("--observations", required=True)
    p.add_argument("--parameters", help="JSON parameter set (kusim schema parameters)")
    p.add_argument("--controls", help="JSON control periods and starting targets (kusim schema controls)")
    p.add_argument("--schedule", help="base schedule (events for controlled wells are replaced)")
    p.add_argument("--restart", metavar="STATE.json")
    p.add_argument("--x", metavar="VECTOR.json", help="JSON list: evaluate at this vector (default: the model as it is)")
    p.add_argument("--method", choices=("adjoint", "fd"), default="adjoint",
                   help="adjoint (about 3 runs; field parameters and controls) or finite differences (any parameter)")
    p.add_argument("--check", type=int, metavar="N", help="compare N random components with finite differences")
    p.add_argument("--upstream", choices=("potential", "pressure"), default="potential")
    p.add_argument("--out", help="file for the whole gradient (JSON)")
    _add_guard_arguments(p)
    p.set_defaults(func=_cmd_gradient)

    p = sub.add_parser("ensemble", help="history matching with an ensemble: ES-MDA or the EnKF")
    p.add_argument("model", help="the base model: the prior mean (unless the prior gives one) and everything not varied")
    p.add_argument("--method", choices=("esmda", "enkf"), required=True,
                   help="esmda: whole history, several passes (recommended); enkf: date by date with restarts")
    p.add_argument("--observations", required=True)
    p.add_argument("--parameters", required=True, help="JSON parameter set: what is uncertain")
    p.add_argument("--prior", required=True, help="JSON prior (kusim schema prior): mean, std and correlation per parameter")
    p.add_argument("--schedule", help="well-control events applied to every member")
    p.add_argument("--members", type=int, default=40, help="ensemble size (default 40)")
    p.add_argument("--seed", type=int, default=11, help="seed: the same seed gives the same ensemble and the same result")
    p.add_argument("--passes", type=int, default=4, help="esmda: number of passes, alpha = passes each (default 4)")
    p.add_argument("--alphas", help="esmda: inflation factors, comma separated, with sum(1/alpha) = 1")
    p.add_argument("--localise", type=float, metavar="FT", help="enkf: Gaspari-Cohn localisation length around wells (ft)")
    p.add_argument("--workers", type=int, default=1, help="processes for the members (default 1)")
    p.add_argument("--out", help="folder for ensemble.json and posterior_mean.json")
    p.add_argument("--write-members", action="store_true", help="also write every posterior member as members/NNNN.json")
    p.add_argument("--write-ensemble", action="store_true",
                   help="also write the prior and posterior parameter vectors (ensemble_parameters.json)")
    p.add_argument("--strict", action="store_true", help="exit with 1 if a posterior member shows signs of trouble")
    _add_guard_arguments(p)
    p.set_defaults(func=_cmd_ensemble)

    p = sub.add_parser("serve", help="serve the commands over HTTP (needs pip install kusim[api]); see kusim.api")
    p.add_argument("--workspace", default="kusim-workspace", help="folder where the documents and results live")
    p.add_argument("--host", default="127.0.0.1", help="address to bind (default: this machine only)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token", help="require this bearer token (or set KUSIM_API_TOKEN)")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds after which a command is stopped (default 900)")
    p.add_argument("--max-concurrent", type=int, default=2, help="commands running at once (default 2); the next gets 429")
    p.add_argument("--max-workers", type=int, default=2, help="ceiling of the workers of an ensemble (default 2)")
    p.add_argument("--max-output-mb", type=float, default=DEFAULT_MAX_OUTPUT_MB, help="ceiling of the output limit")
    p.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS, help="ceiling of the size limit (blocks)")
    p.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="ceiling of the estimated-time limit")
    p.add_argument("--log-level", default="warning", choices=("critical", "error", "warning", "info", "debug"))
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("run", help="run a model")
    p.add_argument("model")
    p.add_argument("--schedule", help="JSON list of well-control events")
    p.add_argument("--out", help="folder for results.json and summary.json (created if needed)")
    p.add_argument("--fields", choices=FIELD_MODES, default="last",
                   help="maps to store in results.json (default: last)")
    p.add_argument("--csv", action="store_true",
                   help="also write wells.csv, perforations.csv, pressure.csv, sw.csv into --out")
    p.add_argument("--solver", choices=METHODS, default="sparse")
    p.add_argument("--upstream", choices=("potential", "pressure"), default="potential")
    p.add_argument("--save-every", type=int, default=1, help="keep a map snapshot every N steps")
    p.add_argument("--strict", action="store_true", help="exit with 1 if the diagnostics show signs of trouble")
    p.add_argument("--restart", metavar="STATE.json", help="continue from a state written by --state-out")
    p.add_argument("--until", type=float, metavar="DAY",
                   help="stop at this day (a multiple of dt, at most t_end) instead of t_end; for staged runs")
    p.add_argument("--state-out", metavar="STATE.json", help="write the state at the end, to restart from")
    _add_guard_arguments(p, output=True)
    p.set_defaults(func=_cmd_run)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        code = args.func(args)
    except _InputError as error:
        _print({"ok": False, "error": str(error), **error.details})
        return EXIT_BAD_INPUT
    return code or 0


if __name__ == "__main__":
    sys.exit(main())
