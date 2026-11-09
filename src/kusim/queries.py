"""Tables from kusim files, and a small declarative query language over them (no code is run).

A plot needs numbers in the right shape: the water cut of one producer against time, the saturation averaged over the layers as
a map, the RMS misfit of every well. Instead of one tool per chart, kusim turns its files into **tables** (``Source``) and lets a
JSON description **query** them (``run_tables``); ``kusim.figure`` then binds the answers into a Plotly figure.

Sources (the kind of a file is recognised from its content, ``kusim tables FILE`` lists the tables and columns):

    model        ``cells`` (block, i, j, k, x, y, depth, sizes, volume, pore_volume, kx, ky, kz, porosity), ``wells``, ``completions``, ``outline`` (the grid boundary in plan view, closed)
    results      ``series`` (time, vector, quantity, well, value: ECLIPSE-style names such as WWCT:P1), ``wells_history``
                 (one row per well and step), ``perforations``, ``snapshots`` (time, block, i, j, k, x, y, depth, pressure, sw, so)
    observations ``observations`` (time, vector, quantity, well, value, sigma, weight)
    residuals    ``residuals`` (the table ``kusim misfit --out`` writes, with quantity and well added)
    summary      ``summary_wells`` (one row per well of a run's ``summary.json``)

A query is a dictionary; its steps run in this order: ``from``, ``join``, ``where``, ``group_by`` + ``aggregate``, ``sort_by``,
``select``, ``limit``, ``pivot``. Anything else is an error that names the key. Lengths are ft, pressure psi, rates STB/day, time days.
"""

from __future__ import annotations

import difflib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import SpecError, _check_keys, _read_json, model_from_dict, results_from_dict

MAX_ROWS = 3_000_000
MAX_FILE_MB = 500
UNITS = {"time": "days", "x": "ft", "y": "ft", "depth": "ft", "dx": "ft", "dy": "ft", "dz": "ft", "volume": "ft3", "pore_volume": "ft3",
         "kx": "md", "ky": "md", "kz": "md", "pressure": "psi", "bhp": "psi", "oil_rate": "STB/day", "water_rate": "STB/day",
         "liquid_rate": "STB/day", "cum_oil": "STB", "cum_water": "STB", "sw": "fraction", "so": "fraction", "porosity": "fraction",
         "water_cut": "fraction"}
QUANTITY_UNITS = {"WBHP": "psi", "WOPR": "STB/day", "WWPR": "STB/day", "WWIR": "STB/day", "WWCT": "fraction", "FPRP": "psi",
                  "FOPT": "STB", "FWPT": "STB", "FWIT": "STB"}

QUERY_KEYS = {"from", "join", "where", "group_by", "aggregate", "sort_by", "select", "limit", "pivot"}
WHERE_OPS = ("==", "!=", "<", "<=", ">", ">=", "in", "not_in")
AGGREGATES = ("mean", "sum", "min", "max", "count", "first", "last", "std", "median", "rms")


# ---------------------------------------------------------------------- helpers
def safe_relative(name, path):
    """A file name that stays inside the working folder (relative, no ``..``, no drive); returns the ``Path``."""
    bad = SpecError(f"{name!r} is not allowed here: a file named inside a plot description must be a relative name in the working "
                    "folder, like 'runs/a/results.json' (no '..', no drive letter)", path, "bad_path")
    if not isinstance(name, str) or not name or ":" in name:
        raise bad
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise bad
    resolved = (Path.cwd() / p).resolve()
    if Path.cwd().resolve() != resolved and Path.cwd().resolve() not in resolved.parents:
        raise bad
    return resolved


def _suggest(name, options):
    close = difflib.get_close_matches(str(name), [str(o) for o in options], n=1)
    return {name: close[0]} if close else {}


def split_vector(vector):
    """``"WWCT:P1"`` -> ``("WWCT", "P1")``; a field vector such as ``"FOPT"`` -> ``("FOPT", None)``."""
    quantity, _, well = str(vector).partition(":")
    return quantity, (well or None)


def _round_times(frame):
    """Floating times are compared after rounding (the same time written by two programs may differ in the last digit)."""
    if "time" in frame.columns and frame["time"].dtype.kind == "f":
        frame = frame.copy()
        frame["time"] = frame["time"].round(9)
    return frame


class Matrix:
    """A table that was pivoted into a grid: ``x`` (column labels), ``y`` (row labels) and ``z`` (rows of values). Refs: ``.x .y .z``."""

    kind = "matrix"

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    @property
    def shape(self):
        return [len(self.y), len(self.x)]

    def part(self, name):
        if name not in ("x", "y", "z"):
            raise KeyError(name)
        return {"x": self.x, "y": self.y, "z": self.z}[name]

    def summary(self, rows=0):
        flat = [v for r in self.z for v in r if v is not None]
        return {"kind": "matrix", "rows_y": len(self.y), "columns_x": len(self.x), "x_range": _range(self.x), "y_range": _range(self.y),
                "z_range": _range(flat), "parts": ["x", "y", "z"]}


def _range(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return [min(values), max(values)] if values else None


def jsonable(frame_or_series):
    """Plain Python lists: NaN and missing become None, NumPy types become numbers."""
    frame = frame_or_series
    if isinstance(frame, pd.Series):
        return [None if (v is None or (isinstance(v, float) and not math.isfinite(v)) or v is pd.NA) else (v.item() if hasattr(v, "item") else v)
                for v in frame.tolist()]
    return {c: jsonable(frame[c]) for c in frame.columns}


def table_summary(table, rows=3):
    if isinstance(table, Matrix):
        return table.summary()
    out = {"kind": "table", "n_rows": int(len(table)), "columns": list(table.columns)}
    if rows and len(table):
        head = table.head(rows)
        out["head"] = [{c: jsonable(head[c])[n] for c in head.columns} for n in range(len(head))]
    return out


def need_columns(frame, columns, path, table):
    for c in columns:
        if c not in frame.columns:
            raise SpecError(f"table {table!r} has no column {c!r}; its columns are {list(frame.columns)}", path, "unknown_column",
                            table=table, columns=list(frame.columns), did_you_mean=_suggest(c, frame.columns))


# ---------------------------------------------------------------------- sources
def detect_kind(data):
    if isinstance(data, dict):
        if "pvt" in data and "grid" in data:
            return "model"
        if "snapshot_times" in data:
            return "results"
        if "wells" in data and "final_time" in data:
            return "summary"
    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = set(data[0])
        if {"observed", "simulated", "residual"} <= keys:
            return "residuals"
        if {"vector", "value", "sigma"} <= keys:
            return "observations"
    return None


def _cell_columns(grid):
    cx = (grid.x_edges[:-1] + grid.x_edges[1:]) / 2
    cy = (grid.y_edges[:-1] + grid.y_edges[1:]) / 2
    return pd.DataFrame({"block": np.arange(1, grid.n_cells + 1), "i": grid.i + 1, "j": grid.j + 1, "k": grid.k + 1,
                         "x": cx[grid.i], "y": cy[grid.j], "depth": grid.depth})


def _target_unit(mode):
    return "psi" if mode == "bhp" else "STB/day"


class Source:
    """The tables of one file. ``tables[name](time_selection)`` builds a table (``time_selection`` only matters for ``snapshots``)."""

    def __init__(self, kind, data, path=""):
        self.kind, self.path = kind, path
        self._cache = {}
        builder = getattr(self, f"_{kind}")
        self._build = builder(data)

    @property
    def names(self):
        return list(self._build)

    def table(self, name, times=None):
        key = (name, None if times is None else tuple(times))
        if key not in self._cache:
            self._cache[key] = _round_times(self._build[name](times))
        return self._cache[key]

    # -- model
    def _model(self, data):
        model = model_from_dict(data)
        g = model.grid

        def cells(_):
            f = _cell_columns(g)
            for c, v in (("dx", g.dx), ("dy", g.dy), ("dz", g.dz), ("volume", g.volume), ("pore_volume", g.volume * model.porosity),
                         ("kx", model.kx), ("ky", model.ky), ("kz", model.kz), ("porosity", model.porosity)):
                f[c] = np.asarray(v, dtype=float)
            return f

        def completions(_):
            rows = []
            for kind, wells, tag in (("producer", model.producers, "P"), ("injector", model.injectors, "I")):
                for n, w in enumerate(wells, start=1):
                    cell = _cell_columns(g).set_index("block")
                    for b in w.blocks:
                        rows.append({"well": f"{tag}{n}", "kind": kind, "block": b, **{c: cell.loc[b, c].item() for c in cell.columns}})
            return pd.DataFrame(rows, columns=["well", "kind", "block", "i", "j", "k", "x", "y", "depth"])

        def wells(_):
            comp = completions(None)
            rows = []
            for kind, ws, tag in (("producer", model.producers, "P"), ("injector", model.injectors, "I")):
                for n, w in enumerate(ws, start=1):
                    mine = comp[comp.well == f"{tag}{n}"]
                    first = mine.iloc[0]
                    rows.append({"well": f"{tag}{n}", "kind": kind, "mode": w.mode, "target": w.target, "target_unit": _target_unit(w.mode),
                                 "n_completions": len(mine), "blocks": ";".join(str(b) for b in w.blocks), "i": int(first.i), "j": int(first.j),
                                 "k_top": int(mine.k.min()), "k_bottom": int(mine.k.max()), "x": float(first.x), "y": float(first.y),
                                 "radius": w.radius, "skin": w.skin})
            return pd.DataFrame(rows, columns=["well", "kind", "mode", "target", "target_unit", "n_completions", "blocks", "i", "j", "k_top",
                                               "k_bottom", "x", "y", "radius", "skin"])

        def outline(_):
            """The plan-view boundary of the grid: five corner points, closed (draw it with a line trace)."""
            x0, x1, y0, y1 = (float(v) for v in (g.x_edges[0], g.x_edges[-1], g.y_edges[0], g.y_edges[-1]))
            return pd.DataFrame({"corner": [1, 2, 3, 4, 5], "x": [x0, x1, x1, x0, x0], "y": [y0, y0, y1, y1, y0]})

        return {"cells": cells, "wells": wells, "completions": completions, "outline": outline}

    # -- results
    def _results(self, data):
        r = results_from_dict(data)
        g = r.grid
        self._times = np.asarray(r.times, dtype=float)

        def series(_):
            w = r.wells
            frames = []
            for name, d in w.groupby("well", sort=False):
                kind = d["kind"].iloc[0]
                names = {"producer": [("WBHP", d.bhp), ("WOPR", d.oil_rate), ("WWPR", d.water_rate), ("WWCT", d.water_cut)],
                         "injector": [("WBHP", d.bhp), ("WWIR", d.water_rate)]}[kind]
                for quantity, values in names:
                    frames.append(pd.DataFrame({"time": d.time.to_numpy(), "vector": f"{quantity}:{name}", "quantity": quantity, "well": name,
                                                "value": values.to_numpy(dtype=float)}))
            for quantity, column, kind in (("FOPT", "cum_oil", "producer"), ("FWPT", "cum_water", "producer"), ("FWIT", "cum_water", "injector")):
                d = w[w.kind == kind]
                if len(d) and column in d.columns:
                    total = d.groupby("time", sort=True)[column].sum()
                    frames.append(pd.DataFrame({"time": total.index.to_numpy(), "vector": quantity, "quantity": quantity, "well": None,
                                                "value": total.to_numpy(dtype=float)}))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "vector", "quantity", "well", "value"])

        def snapshots(times):
            chosen = _select_times(r.times, times)
            if len(chosen) * g.n_cells > MAX_ROWS:
                raise SpecError(f"the snapshots table would have {len(chosen) * g.n_cells} rows (over {MAX_ROWS}): select fewer times with "
                                "where time == ... (a number, 'first' or 'last')", "", "too_large")
            if not len(chosen):
                return pd.DataFrame(columns=["time", "block", "i", "j", "k", "x", "y", "depth", "pressure", "sw", "so"])
            base = _cell_columns(g)
            frames = []
            for s in chosen:
                f = base.copy()
                f.insert(0, "time", float(r.times[s]))
                f["pressure"], f["sw"] = r.pressure[s], r.sw[s]
                f["so"] = 1.0 - f["sw"]
                frames.append(f)
            return pd.concat(frames, ignore_index=True)

        return {"series": series, "wells_history": lambda _: r.wells.copy(), "perforations": lambda _: r.perforations.copy(),
                "snapshots": snapshots}

    # -- observations, residuals, summary
    def _observations(self, data):
        def observations(_):
            f = pd.DataFrame(data)
            parts = [split_vector(v) for v in f["vector"]]
            f.insert(2, "quantity", [p[0] for p in parts])
            f.insert(3, "well", [p[1] for p in parts])
            return f

        return {"observations": observations}

    def _residuals(self, data):
        def residuals(_):
            f = pd.DataFrame(data)
            parts = [split_vector(v) for v in f["vector"]]
            f.insert(2, "quantity", [p[0] for p in parts])
            f.insert(3, "well", [p[1] for p in parts])
            return f

        return {"residuals": residuals}

    def _summary(self, data):
        def summary_wells(_):
            rows = [{"well": name, **{k: v for k, v in w.items()}} for name, w in data["wells"].items()]
            return pd.DataFrame(rows)

        return {"summary_wells": summary_wells}

    def describe(self, rows=2):
        """For each table: columns, row count, the first rows, units (and the quantities of a long table of vectors)."""
        out = {}
        for name in self.names:
            table = self.table(name, ["last"] if name == "snapshots" else None)
            entry = table_summary(table, rows)
            if name == "snapshots":
                entry["n_rows"] = int(len(self._times) * len(table))
                entry["saved_times"] = {"n": int(len(self._times)), "first": float(self._times[0]), "last": float(self._times[-1])}
                entry["note"] = "one row per block and saved time; select times with where time == 'last' (or a number). The head shows the last time"
            entry["units"] = {c: UNITS[c] for c in table.columns if c in UNITS}
            if "quantity" in table.columns:
                entry["quantities"] = sorted(table["quantity"].dropna().unique().tolist())
                entry["quantity_units"] = {q: QUANTITY_UNITS[q] for q in entry["quantities"] if q in QUANTITY_UNITS}
            out[name] = entry
        return out

    _times = np.zeros(0)


def _select_times(times, wanted):
    """Indices of the saved times that a ``where`` on the column ``time`` can keep (``wanted`` None: all)."""
    n = len(times)
    if wanted is None:
        return list(range(n))
    keep = set()
    for w in wanted:
        if w == "last":
            keep.add(n - 1)
        elif w == "first":
            keep.add(0)
        else:
            keep |= {s for s in range(n) if math.isclose(float(times[s]), float(w), abs_tol=1e-9)}
    return sorted(keep)


def load_source(name, file, path):
    """Read a file and make its tables; ``name`` and ``path`` are for messages."""
    resolved = safe_relative(file, path)
    if not resolved.is_file():
        folder = resolved.parent
        nearby = sorted(p.name for p in folder.iterdir() if p.is_file())[:20] if folder.is_dir() else []
        raise SpecError(f"source {name!r}: file not found: {file}", path, "file_not_found", file=file, files_nearby=nearby)
    if resolved.stat().st_size > MAX_FILE_MB * 1_000_000:
        raise SpecError(f"source {name!r}: {file} is over {MAX_FILE_MB} MB", path, "too_large")
    return source_from_data(_read_json(resolved), file, path)


def source_from_data(data, file="", path=""):
    kind = detect_kind(data)
    if kind is None:
        raise SpecError(f"{file or 'this file'} is not a file that kusim can make tables from. Give a model, the results.json or summary.json "
                        "of a run, an observations file or the table of 'kusim misfit --out'", path, "bad_source", file=file)
    return Source(kind, data, file)


# ---------------------------------------------------------------------- queries
class TableSet:
    """The named sources and the tables derived so far; ``get`` resolves ``source.table`` or the name of a derived table."""

    def __init__(self, sources):
        self.sources, self.derived = sources, {}

    def available(self):
        return sorted(self.derived) + [f"{s}.{t}" for s, src in self.sources.items() for t in src.names]

    def get(self, name, path, times=None):
        if name in self.derived:
            return self.derived[name]
        head, dot, rest = str(name).partition(".")
        if dot and head in self.sources:
            source = self.sources[head]
            if rest in source.names:
                return source.table(rest, times)
            hint = ""
            if source.kind == "model" and rest in ("series", "snapshots", "perforations", "wells_history"):
                hint = (f" ({head!r} is a MODEL file; a run table needs the results.json of a run: kusim run MODEL --out DIR, then use DIR/results.json)")
            raise SpecError(f"source {head!r} ({source.kind}) has no table {rest!r}; its tables are {source.names}{hint}", path, "unknown_table",
                            available=[f"{head}.{t}" for t in source.names], did_you_mean=_suggest(rest, source.names))
        raise SpecError(f"there is no table or source table {name!r}; available: {self.available()}", path, "unknown_table",
                        available=self.available(), did_you_mean=_suggest(name, self.available()))


def _time_conditions(where):
    """The values that conditions on ``time`` select, if every condition on time is an equality or 'in' (so that fewer snapshots are built)."""
    wanted = None
    for c in where:
        if isinstance(c, dict) and c.get("column") == "time":
            if c.get("op") == "==":
                wanted = [c.get("value")]
            elif c.get("op") == "in" and isinstance(c.get("value"), list):
                wanted = list(c["value"])
            else:
                return None
    return wanted


def _scalar(frame, column, value):
    if value in ("first", "last"):
        series = frame[column]
        if not len(series):
            return value
        return series.min() if value == "first" else series.max()
    return value


def _apply_where(frame, where, path, table):
    for n, c in enumerate(where):
        here = f"{path}.where.{n}"
        _check_keys("a where condition", c, {"column", "op", "value"}, {"column", "op", "value"}, path=here)
        need_columns(frame, [c["column"]], f"{here}.column", table)
        op, col = c["op"], frame[c["column"]]
        if op not in WHERE_OPS:
            raise SpecError(f"where.op must be one of {list(WHERE_OPS)}, got {op!r}", f"{here}.op", "bad_value", allowed=list(WHERE_OPS))
        value = c["value"]
        try:
            if op in ("in", "not_in"):
                if not isinstance(value, list):
                    raise SpecError(f"where with op {op!r} needs a list value", f"{here}.value", "bad_value")
                mask = pd.Series(False, index=frame.index)
                for v in value:
                    v = _scalar(frame, c["column"], v)
                    mask |= _equal(col, v)
                mask = mask if op == "in" else ~mask
            else:
                value = _scalar(frame, c["column"], value)
                mask = {"==": lambda: _equal(col, value), "!=": lambda: ~_equal(col, value), "<": lambda: col < value, "<=": lambda: col <= value,
                        ">": lambda: col > value, ">=": lambda: col >= value}[op]()
        except TypeError:
            raise SpecError(f"cannot compare column {c['column']!r} ({col.dtype}) with {value!r}", f"{here}.value", "bad_value") from None
        frame = frame[mask]
    return frame


def _equal(col, value):
    if col.dtype.kind == "f" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.Series(np.isclose(col.to_numpy(dtype=float), float(value), rtol=0, atol=1e-9), index=col.index)
    return col == value


def _aggregate(frame, group_by, aggregate, path, table):
    for n, (out, spec) in enumerate(aggregate.items()):
        here = f"{path}.aggregate.{out}"
        _check_keys("an aggregate", spec, {"fn", "column", "weight"}, {"fn"}, path=here)
        if spec["fn"] not in AGGREGATES:
            raise SpecError(f"aggregate fn must be one of {list(AGGREGATES)}, got {spec['fn']!r}", f"{here}.fn", "bad_value", allowed=list(AGGREGATES))
        if spec["fn"] != "count" and "column" not in spec:
            raise SpecError(f"aggregate {out!r} needs a 'column'", f"{here}.column", "missing_key")
        need_columns(frame, [c for c in (spec.get("column"), spec.get("weight")) if c], here, table)
        if "weight" in spec and spec["fn"] != "mean":
            raise SpecError("'weight' only works with fn 'mean'", f"{here}.weight", "bad_value")
    keys = list(group_by)
    work = frame.copy()
    parts = {}
    grouped = work.groupby(keys, sort=True, dropna=False) if keys else None
    for out, spec in aggregate.items():
        fn, column, weight = spec["fn"], spec.get("column"), spec.get("weight")
        if fn == "count":
            part = grouped.size() if keys else pd.Series([len(work)])
        elif weight:
            work["_wx"] = work[column] * work[weight]
            if keys:
                num, den = work.groupby(keys, sort=True, dropna=False)["_wx"].sum(), work.groupby(keys, sort=True, dropna=False)[weight].sum()
            else:
                num, den = pd.Series([work["_wx"].sum()]), pd.Series([work[weight].sum()])
            part = num / den
        elif fn == "rms":
            work["_sq"] = work[column] ** 2
            part = (work.groupby(keys, sort=True, dropna=False)["_sq"].mean() if keys else pd.Series([work["_sq"].mean()])) ** 0.5
        else:
            pandas_fn = {"first": "first", "last": "last", "std": "std", "median": "median"}.get(fn, fn)
            part = grouped[column].agg(pandas_fn) if keys else pd.Series([getattr(work[column], pandas_fn)()])
        parts[out] = part
    result = pd.DataFrame(parts)
    return result.reset_index() if keys else result


def run_query(query, tables, path, name):
    """One table (a ``DataFrame``, or a ``Matrix`` after ``pivot``) from a query dictionary."""
    _check_keys(f"the query {name!r}", query, QUERY_KEYS, {"from"}, path=path)
    where = query.get("where", [])
    if isinstance(where, dict):
        where = [where]
    if not isinstance(where, list):
        raise SpecError("where must be a list of conditions", f"{path}.where", "bad_value")
    frame = tables.get(query["from"], f"{path}.from", times=_time_conditions(where))
    if isinstance(frame, Matrix):
        raise SpecError(f"{query['from']!r} is a pivoted matrix: queries start from tables (pivot as the last step)", f"{path}.from", "bad_value")
    if "join" in query:
        join = query["join"]
        _check_keys("a join", join, {"table", "on", "how", "suffix"}, {"table", "on"}, path=f"{path}.join")
        right = tables.get(join["table"], f"{path}.join.table")
        if isinstance(right, Matrix):
            raise SpecError("cannot join a pivoted matrix", f"{path}.join.table", "bad_value")
        on = [join["on"]] if isinstance(join["on"], str) else list(join["on"])
        need_columns(frame, on, f"{path}.join.on", query["from"])
        need_columns(right, on, f"{path}.join.on", join["table"])
        how = join.get("how", "inner")
        if how not in ("inner", "left"):
            raise SpecError("join.how must be 'inner' or 'left'", f"{path}.join.how", "bad_value", allowed=["inner", "left"])
        frame = _round_times(frame).merge(_round_times(right), on=on, how=how, suffixes=("", join.get("suffix", "_right")))
    frame = _apply_where(frame, where, path, query["from"])
    group_by = query.get("group_by", [])
    group_by = [group_by] if isinstance(group_by, str) else list(group_by)
    if group_by and "aggregate" not in query:
        raise SpecError("group_by needs an aggregate (for example {\"mean_sw\": {\"fn\": \"mean\", \"column\": \"sw\"}})", f"{path}.group_by", "missing_key")
    if "aggregate" in query:
        if not isinstance(query["aggregate"], dict) or not query["aggregate"]:
            raise SpecError("aggregate must be a non-empty object: output name -> {fn, column}", f"{path}.aggregate", "bad_value")
        need_columns(frame, group_by, f"{path}.group_by", query["from"])
        frame = _aggregate(frame, group_by, query["aggregate"], path, query["from"])
    if "sort_by" in query:
        keys = [query["sort_by"]] if isinstance(query["sort_by"], str) else list(query["sort_by"])
        names = [k[1:] if k.startswith("-") else k for k in keys]
        need_columns(frame, names, f"{path}.sort_by", name)
        frame = frame.sort_values(names, ascending=[not k.startswith("-") for k in keys], kind="stable")
    if "select" in query:
        select = [query["select"]] if isinstance(query["select"], str) else list(query["select"])
        need_columns(frame, select, f"{path}.select", name)
        frame = frame[select]
    if "limit" in query:
        if not isinstance(query["limit"], int) or isinstance(query["limit"], bool) or query["limit"] < 0:
            raise SpecError("limit must be a whole number", f"{path}.limit", "bad_value")
        frame = frame.head(query["limit"])
    frame = frame.reset_index(drop=True)
    if len(frame) > MAX_ROWS:
        raise SpecError(f"table {name!r} has {len(frame)} rows (over {MAX_ROWS}): filter or aggregate it", path, "too_large")
    if "pivot" in query:
        frame = _pivot(frame, query["pivot"], f"{path}.pivot", name)
    return frame


def _pivot(frame, pivot, path, name):
    _check_keys("a pivot", pivot, {"rows", "columns", "values"}, {"rows", "columns", "values"}, path=path)
    need_columns(frame, [pivot["rows"], pivot["columns"], pivot["values"]], path, name)
    try:
        wide = frame.pivot(index=pivot["rows"], columns=pivot["columns"], values=pivot["values"])
    except ValueError:
        raise SpecError(f"pivot of {name!r}: some (rows, columns) pairs occur more than once. Aggregate first: group_by "
                        f"[{pivot['rows']!r}, {pivot['columns']!r}] with an aggregate of {pivot['values']!r} (for example fn 'mean'), or "
                        "filter to one layer or one time", path, "pivot_duplicates") from None
    wide = wide.sort_index().sort_index(axis=1)
    z = [jsonable(pd.Series(row)) for row in wide.to_numpy(dtype=float)]
    return Matrix(jsonable(pd.Series(wide.columns)), jsonable(pd.Series(wide.index)), z)


def run_tables(sources, queries, path="tables"):
    """Run the queries in order; returns the ``TableSet`` (``.derived`` has the tables)."""
    tables = TableSet(sources)
    if not isinstance(queries, dict):
        raise SpecError("tables must be an object: table name -> query", path, "not_an_object")
    for name, query in queries.items():
        if "." in name or not name:
            raise SpecError(f"a table name may not be empty or contain '.', got {name!r}", f"{path}.{name}", "bad_value")
        if name in sources:
            raise SpecError(f"{name!r} is already the name of a source", f"{path}.{name}", "bad_value")
        tables.derived[name] = run_query(query, tables, f"{path}.{name}", name)
    return tables
