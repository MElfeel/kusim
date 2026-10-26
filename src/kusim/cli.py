"""Command line: run kusim from a shell, a script in any language, or an agent, with JSON files in and out.

    kusim template -o model.json                    write an example model (a small layered waterflood)
    kusim convert INPUT.DAT -o model.json           turn a legacy input file into a JSON model
    kusim validate model.json                       check a model without running it
    kusim run model.json --out results/             run it; writes results/results.json and prints a summary
    kusim run model.json --schedule schedule.json   ... with well-control changes in time
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
from .solver import METHODS
from .spec import (FIELD_MODES, _clean, load_model, load_schedule, model_to_dict, run_summary, save_model,
                   save_results)

EXIT_BAD_INPUT = 2
EXIT_TROUBLE = 1


class _InputError(Exception):
    """Something is wrong with what the caller gave us."""


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


def _load(path, loader, what):
    try:
        return loader(path)
    except FileNotFoundError:
        raise _InputError(f"{what} file not found: {path}") from None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise _InputError(f"{what} {path} is not valid: {error}") from None


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
    _print(info)


def _cmd_run(args):
    from .schedule import advance
    from .simulator import Simulator

    model = _load(args.model, load_model, "model")
    schedule = _load(args.schedule, load_schedule, "schedule") if args.schedule else None
    if schedule is not None:
        _check_schedule(model, schedule, args.schedule)
    sim = Simulator(model, linear_solver=args.solver, save_every=args.save_every, upstream=args.upstream)
    if schedule is None:
        sim.run()
    else:
        advance(sim, model.t_end, schedule)
    results = sim.results()
    summary = run_summary(results, sim.diagnostics())
    summary.update(n_steps=sim.step_number, solver=args.solver, upstream=args.upstream, kusim=__version__)
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
    p.set_defaults(func=_cmd_validate)

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
    p.set_defaults(func=_cmd_run)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        code = args.func(args)
    except _InputError as error:
        _print({"ok": False, "error": str(error)})
        return EXIT_BAD_INPUT
    return code or 0


if __name__ == "__main__":
    sys.exit(main())
