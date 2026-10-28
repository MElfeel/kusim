"""Command line: run kusim from a shell, a script in any language, or an agent, with JSON files in and out.

    kusim template -o model.json                    write an example model (a small layered waterflood)
    kusim convert INPUT.DAT -o model.json           turn a legacy input file into a JSON model
    kusim validate model.json                       check a model without running it
    kusim run model.json --out results/             run it; writes results/results.json and prints a summary
    kusim run model.json --schedule schedule.json   ... with well-control changes in time
    kusim run model.json --until 200 --state-out s.json    run to day 200 and keep the state;
    kusim run model.json --restart s.json --schedule x.json   ... continue from it (changing wells on the way)
    kusim misfit model.json --observations obs.json   run it and compare with data: objective, rms, worst residuals
    kusim edit model.json --set kx=150 -o new.json  change values by path (or --patch file, --in-place)
    kusim diff a.json b.json                        what differs between two models, by path
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
import sys
from pathlib import Path

from . import __version__
from .schema import SCHEMAS, get_schema
from .solver import METHODS
from .spec import (FIELD_MODES, SpecError, _clean, load_model, load_schedule, load_state, _write_json, model_to_dict, run_summary, save_model,
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
        "producers": [{"name": f"P{n}", "blocks": w.blocks, "mode": w.mode, "target": w.target}
                      for n, w in enumerate(model.producers, start=1)],
        "injectors": [{"name": f"I{n}", "blocks": w.blocks, "mode": w.mode, "target": w.target}
                      for n, w in enumerate(model.injectors, start=1)],
    }


def _details(error):
    """The machine-readable part of a SpecError (code, path, hints) without the message."""
    return {k: v for k, v in error.as_dict().items() if k != "error"}


def _load(path, loader, what):
    try:
        return loader(path)
    except FileNotFoundError:
        raise _InputError(f"{what} file not found: {path}") from None
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
    _print(info)


def _cmd_edit(args):
    from .edit import apply_edits, parse_value
    from .spec import _read_json

    data = _load(args.model, _read_json, "model")
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
    _print({"ok": True, "written": str(out), "changes": changes, **_describe(model)})


def _cmd_diff(args):
    from .edit import diff_models

    a, b = _load(args.a, load_model, "model"), _load(args.b, load_model, "model")
    changes = diff_models(a, b)
    _print({"ok": True, "identical": not changes, "n_differences": len(changes), "differences": changes})


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
    from .simulator import Simulator
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
    try:
        record = run_to(model, observations.times, schedule, observations.vectors, state, upstream=args.upstream)
    except ValueError as error:
        raise _InputError(f"cannot run the model: {error}", code="invalid_value") from None
    reached = set(record.table.TIME.round(9)) if len(record.table) else set()
    have = observations.table[observations.table.time.round(9).isin(reached)]
    missing = observations.table[~observations.table.time.round(9).isin(reached)]
    table = misfit(type(observations)(have), record.table) if len(have) else None
    healthy = record.ok and not len(missing) and record.diagnostics["n_saturation_clipped"] == 0 \
        and record.diagnostics["n_backflow_well_steps"] == 0
    out = {"ok": True, "n_observations": len(observations.table), "n_compared": 0 if table is None else len(table),
           "n_missing": len(missing), "final_time": record.state.time, "diagnostics": record.diagnostics,
           "healthy": healthy}
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
    if schedule is None and not args.restart and args.until is None:
        sim.run()
    else:
        try:
            advance(sim, until, schedule)
        except ValueError as error:
            raise _InputError(f"cannot advance to day {until}: {error}", code="invalid_value") from None
    results = sim.results()
    summary = run_summary(results, sim.diagnostics())
    summary.update(final_time=sim.time, start_time=start, n_steps=sim.step_number, solver=args.solver,
                   upstream=args.upstream, kusim=__version__)
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

    p = sub.add_parser("validate", help="check a model (and a schedule) without running")
    p.add_argument("model")
    p.add_argument("--schedule", help="JSON list of well-control events")
    p.add_argument("--restart", metavar="STATE.json", help="also check that this state fits the model")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("edit", help="change values of a model by path (producers.0.target=2500) and write it")
    p.add_argument("model")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", dest="assignments",
                   help="a value as JSON or text; repeatable; paths count list positions from 0")
    p.add_argument("--patch", help="JSON file with an object merged into the model (lists are replaced)")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("-o", "--output", help="file to write")
    target.add_argument("--in-place", action="store_true", help="overwrite the model file")
    p.set_defaults(func=_cmd_edit)

    p = sub.add_parser("diff", help="list what differs between two models, by path")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=_cmd_diff)

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
    p.add_argument("--strict", action="store_true", help="exit with 1 if the run shows signs of trouble")
    p.set_defaults(func=_cmd_misfit)

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
