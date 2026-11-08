"""Compare two finished runs (the ``summary.json`` that ``kusim run --out`` writes), well by well.

An agent that is asked "run both cases and compare" should not have to read two large result files and subtract by hand.
``compare_summaries`` returns, per well and for the field, the two values, the change and the change in percent, and a list of
``notes`` for the things a careful reader would check first: identical numbers (did the change act at all?), runs that are not
healthy, runs of different length, runs on grids of different size (a difference is then a discretisation effect, not a model
change).
"""

from __future__ import annotations

import json
from pathlib import Path

WELL_VALUES = ("mode", "bhp", "oil_rate", "water_rate", "liquid_rate", "water_cut", "cum_oil", "cum_water")
RELATIVE_TOLERANCE = 1e-9


def load_summary(path):
    """The summary of a run: ``path`` is a run folder (with ``summary.json``) or the summary file itself."""
    path = Path(path)
    file = path / "summary.json" if path.is_dir() else path
    if not file.exists():
        raise FileNotFoundError(file)
    data = json.loads(file.read_text(encoding="utf8"))
    if not isinstance(data, dict) or "wells" not in data:
        raise ValueError(f"{file} is not the summary of a run (it has no 'wells'); give a folder written by 'kusim run --out'")
    return data


def _pair(a, b):
    """``{"a", "b"}`` and, for two numbers, ``change`` and ``change_percent`` (relative to a)."""
    out = {"a": a, "b": b}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        out["change"] = b - a
        if a != 0:
            out["change_percent"] = 100.0 * (b - a) / abs(a)
    return out


def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= RELATIVE_TOLERANCE * max(abs(a), abs(b), 1.0)
    return a == b


def compare_summaries(a, b):
    """Differences between two run summaries (dictionaries): ``wells``, ``field``, ``same_numbers`` and ``notes``."""
    notes, wells, identical = [], {}, True
    for name in sorted(set(a["wells"]) | set(b["wells"])):
        wa, wb = a["wells"].get(name), b["wells"].get(name)
        if wa is None or wb is None:
            notes.append(f"well {name} is only in run {'b' if wa is None else 'a'}")
            identical = False
            continue
        row = {key: _pair(wa.get(key), wb.get(key)) for key in WELL_VALUES if wa.get(key) is not None or wb.get(key) is not None}
        wells[name] = row
        identical &= all(_same(v["a"], v["b"]) for v in row.values())
    field = {}
    for key in ("cum_oil", "cum_water"):
        total = [sum(w[key] or 0.0 for w in s["wells"].values() if w.get("kind") == "producer") for s in (a, b)]
        field[f"producers_{key}"] = _pair(*total)
    injected = [sum(w["cum_water"] or 0.0 for w in s["wells"].values() if w.get("kind") == "injector") for s in (a, b)]
    field["injected_cum_water"] = _pair(*injected)
    run = {"final_time": _pair(a.get("final_time"), b.get("final_time")), "n_cells": _pair(a.get("n_cells"), b.get("n_cells")),
           "n_steps": _pair(a.get("n_steps"), b.get("n_steps")), "healthy": _pair(a.get("healthy"), b.get("healthy"))}
    if identical:
        notes.append("the two runs gave identical numbers: check that your change acts. For example, the minimum bottom-hole pressure "
                     "(min_bhp) of a producer that is in mode 'bhp' does nothing, its pressure is 'target'; use "
                     "'kusim diff' on the two models and read the control mode of each well in the summaries (mode, bhp)")
    for label, summary in (("a", a), ("b", b)):
        if summary.get("healthy") is False:
            notes.append(f"run {label} is not healthy (see its diagnostics): do not use its numbers before you have fixed that")
    if not _same(a.get("final_time"), b.get("final_time")):
        notes.append("the runs end at different times: cumulative volumes are not comparable")
    if a.get("n_cells") != b.get("n_cells"):
        notes.append("the runs use grids of different size: a difference is a discretisation effect (and the time step may differ), "
                     "not a change of the physics; cumulative volumes and rates are the quantities to compare")
    return {"same_numbers": bool(identical), "run": run, "wells": wells, "field": field, "notes": notes}
