"""Load historical well data from an Excel workbook (or a CSV file) as kusim observations, and optionally as well controls.

    imported = read_history("field_data.xlsx", model=model, dt=model.dt, wells={"PROD-A": "P1"})
    imported.observations        # Observations: time, vector, value, sigma, weight (the format of history matching)
    imported.schedule            # Schedule of well controls made from the measured rates (with controls=True)
    imported.report              # what was read, dropped, converted and assumed

The sheet may be laid out in one of three ways (``layout="auto"`` looks at the headers):

``by_well``  one row per date and well, one column per quantity (the usual production database export)::

                 Date        Well   Oil rate (STB/d)  Water rate (STB/d)  BHP (psi)
                 2020-02-01  P1     2950              35                  1620

``wide``     one row per date, one column per well and quantity: ``WBHP:P1``, ``P1 BHP``, ``BHP P1``, ``P1_oil_rate``.

``long``     one row per measurement: ``time`` (or ``date``), ``vector`` (``WBHP:P1``, or a quantity with a ``well`` column),
             ``value`` and optionally ``sigma`` and ``weight``.

Time is a column of days since the start (``time``/``days``) or of dates (``date``; days are counted from ``start_date``, by
default the first date). Units come from the header (``BHP (bar)``) or the ``*_unit`` options and are converted to
kusim's oilfield units. Sigma (the standard deviation of the measurement error) comes from a ``sigma`` column, from the
``sigma`` option (``{"WBHP": 25, "WOPR": "5%"}``) or from defaults, which the report lists.

Observations and controls must lie on kusim's time step: with ``dt`` (or a model) the data are grouped into intervals of
``every`` days (default ``dt``) and the mean (or the last value) of each interval is an observation at the end of the
interval. Controls made from rates act from the start of the interval in which the rate was measured.
Nothing is dropped silently: ``report["dropped"]`` counts the rows left out and names the first few (Excel row numbers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .history import Observations
from .schedule import ControlEvent, Schedule
from .spec import SpecError

PRESSURE_UNITS = {"psi": 1.0, "psia": 1.0, "bar": 14.5037738, "bara": 14.5037738, "kpa": 0.145037738, "mpa": 145.037738,
                  "atm": 14.6959488, "kgf/cm2": 14.2233433, "kg/cm2": 14.2233433}
RATE_UNITS = {"stb/d": 1.0, "stb/day": 1.0, "bbl/d": 1.0, "bbl/day": 1.0, "bpd": 1.0, "bopd": 1.0, "bwpd": 1.0,
              "m3/d": 6.28981077, "m3/day": 6.28981077, "sm3/d": 6.28981077, "sm3/day": 6.28981077}
WATER_CUT_UNITS = {"fraction": 1.0, "frac": 1.0, "-": 1.0, "%": 0.01, "percent": 0.01, "pct": 0.01}
QUANTITIES = {
    "bhp": ["flowing bottom hole pressure", "bottom hole pressure", "bottomhole pressure", "bottom-hole pressure", "bhp",
            "bhfp", "pwf", "wbhp", "pressure"],
    "oil_rate": ["oil production rate", "oil rate", "oil production", "oil prod", "oil", "qo", "wopr", "orat"],
    "water_rate": ["water production rate", "water rate", "water production", "water prod", "water", "qw", "wwpr", "wrat"],
    "liquid_rate": ["liquid production rate", "total liquid rate", "liquid rate", "liquid", "qliq", "ql", "wlpr", "lrat"],
    "water_cut": ["water cut", "watercut", "wwct", "wcut", "wc", "fw"],
    "water_inj": ["water injection rate", "water injection", "water inj rate", "water inj", "injection rate", "injection",
                  "winj", "wwir", "qwi", "wir"],
}
UNIT_OF = {"bhp": "pressure", "water_cut": "water_cut"}          # everything else is a rate
VECTOR_PREFIXES = ("WBHP", "WOPR", "WWPR", "WWCT", "WWIR")
PRODUCER_VECTORS = {"bhp": "WBHP", "oil_rate": "WOPR", "water_rate": "WWPR", "water_cut": "WWCT"}
INJECTOR_VECTORS = {"bhp": "WBHP", "water_inj": "WWIR", "water_rate": "WWIR"}
DEFAULT_SIGMA = {"WBHP": 25.0, "WWCT": 0.02, "WOPR": "5%", "WWPR": "5%", "WWIR": "5%"}
TIME_NAMES = {"time", "days", "day", "t", "elapsed", "elapsed time", "elapsed days", "time since start"}
DATE_NAMES = {"date", "datetime", "timestamp", "time stamp", "report date", "measurement date"}
WELL_NAMES = {"well", "well name", "wellname", "well id", "name", "uwi"}
VECTOR_NAMES = {"vector", "variable", "quantity", "measurement", "keyword", "property"}
VALUE_NAMES = {"value", "measurement value", "measured value"}
SIGMA_NAMES = {"sigma", "std", "stdev", "std dev", "uncertainty", "error", "standard deviation"}
WEIGHT_NAMES = {"weight"}
ECLIPSE_VECTOR = re.compile(r"^(WBHP|WOPR|WWPR|WWCT|WWIR)[:\s_.\-]+(.+)$", re.I)


@dataclass
class HistoryImport:
    observations: Observations
    schedule: Schedule
    report: dict = field(default_factory=dict)


def _fail(where, code, message, **details):
    raise SpecError(message, where, code, **details)


def _split_unit(header):
    """('BHP', 'psi') from 'BHP (psi)' or 'BHP [psi]'."""
    text = str(header).strip()
    match = re.search(r"[\(\[]([^\)\]]*)[\)\]]\s*$", text)
    if not match:
        return text, None
    return text[:match.start()].strip(), match.group(1).strip().lower() or None


def _normal(text):
    return re.sub(r"\s+", " ", re.sub(r"[_]+", " ", str(text)).strip().lower())


def _quantity_of(name):
    """The quantity key of a header such as 'Oil rate', or None."""
    key = _normal(name)
    for quantity, aliases in QUANTITIES.items():
        if key in aliases:
            return quantity
    return None


def _quantity_and_well(header):
    """(quantity, well) of a wide-layout header like 'P1 BHP', 'BHP P1' or 'WBHP:P1'; None when it is not one."""
    match = ECLIPSE_VECTOR.match(header)
    if match:
        prefix = match.group(1).upper()
        quantity = next(q for q, v in {**PRODUCER_VECTORS, "water_inj": "WWIR"}.items() if v == prefix)
        return quantity, match.group(2).strip()
    aliases = sorted(((a, q) for q, al in QUANTITIES.items() for a in al), key=lambda x: -len(x[0]))
    for alias, quantity in aliases:
        pattern = r"[\s_\-]*".join(re.escape(w) for w in alias.split())
        for regex in (rf"^(?P<w>.+?)[\s_:\-]+{pattern}$", rf"^{pattern}[\s_:\-]+(?P<w>.+)$"):
            found = re.match(regex, header, re.I)
            if found:
                return quantity, found.group("w").strip()
    return None


def _convert(unit_kind, unit, where):
    table = {"pressure": PRESSURE_UNITS, "rate": RATE_UNITS, "water_cut": WATER_CUT_UNITS}[unit_kind]
    if unit.lower() not in table:
        _fail(where, "unknown_unit", f"the {unit_kind} unit {unit!r} is not known", allowed=sorted(table),
              hint="convert the column to one of these units, or give the unit in the header, e.g. 'BHP (bar)'")
    return table[unit.lower()]


def _parse_sigma(value, where):
    """('abs', x) or ('rel', fraction) from 25, '25' or '5%'."""
    if isinstance(value, str) and value.strip().endswith("%"):
        return "rel", float(value.strip()[:-1]) / 100.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(where, "bad_value", f"sigma {value!r} is not a number or a percentage (25 or '5%')")
    if number <= 0:
        _fail(where, "bad_value", f"sigma must be positive, got {value!r}")
    return "abs", number


def _prefix_of(key, where):
    if str(key).upper() in VECTOR_PREFIXES:
        return str(key).upper()
    quantity = _quantity_of(key)
    if quantity in ("water_rate",):
        return "WWPR"
    if quantity is not None:
        return {**PRODUCER_VECTORS, "water_inj": "WWIR"}.get(quantity)
    _fail(where, "bad_value", f"{key!r} is not a vector of the sigma option", allowed=list(VECTOR_PREFIXES))


def _read_table(path, sheet, header_row):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=header_row - 1), "csv"
        if path.suffix.lower() not in (".xlsx", ".xlsm"):
            _fail(path.name, "unsupported_file", f"{path.suffix or 'this file'} is not read: use .xlsx, .xlsm or .csv",
                  hint=".xls (old Excel) files: save as .xlsx first")
        try:
            book = pd.ExcelFile(path, engine="openpyxl")
        except ImportError:
            _fail(path.name, "missing_dependency", "reading Excel needs openpyxl", hint="pip install 'kusim[excel]'")
        names = book.sheet_names
        if sheet is None:
            sheet = names[0]
        elif sheet not in names:
            _fail(path.name, "unknown_sheet", f"the workbook has no sheet {sheet!r}", allowed=names,
                  did_you_mean=[n for n in names if n.lower() == str(sheet).lower()])
        return book.parse(sheet, header=header_row - 1), sheet
    except FileNotFoundError:
        _fail(str(path), "file_not_found", f"cannot read {path}")
    except SpecError:
        raise
    except Exception as error:                      # a damaged or password-protected workbook, an empty csv, ...
        _fail(path.name, "unreadable_file", f"{path.name} cannot be read: {error}")


def read_history(path, *, sheet=None, layout="auto", header_row=1, start_date=None, dayfirst=False, model=None, dt=None,
                 every=None, align="mean", t_end=None, wells=None, vectors=None, pressure_unit="psi", rate_unit="stb/d",
                 water_cut_unit="fraction", sigma=None, controls=False, producer_control="liquid_rate",
                 skip_wells=None) -> HistoryImport:
    """Read a workbook (see the module docstring). ``model`` gives the well names, the well kinds, ``dt`` and ``t_end``;
    ``wells`` maps the names in the sheet to kusim's (``{"PROD-A": "P1"}``); ``vectors`` keeps only those vectors
    (``["WBHP", "WWCT"]``). ``controls=True`` also returns a schedule made from the measured rates. ``skip_wells`` lists wells of
    the sheet to leave out on purpose (they are counted in ``dropped`` and named in ``skipped_wells``). Two wells of the sheet may
    never share one kusim name: that would mix the data of two wells."""
    if layout not in ("auto", "long", "by_well", "wide"):
        _fail("layout", "bad_value", f"unknown layout {layout!r}", allowed=["auto", "long", "by_well", "wide"])
    if align not in ("mean", "last"):
        _fail("align", "bad_value", f"unknown align {align!r}", allowed=["mean", "last"])
    if producer_control not in ("liquid_rate", "oil_rate"):
        _fail("producer_control", "bad_value", "producer_control must be liquid_rate or oil_rate")
    name = Path(path).name
    frame, sheet_used = _read_table(path, sheet, header_row)
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    frame.columns = [str(c).strip() for c in frame.columns]
    rows = {i: header_row + 1 + n for n, i in enumerate(frame.index)}          # frame index -> Excel row
    report = {"file": name, "sheet": sheet_used, "rows_read": len(frame), "dropped": {}, "warnings": [], "defaults_used": {},
              "units": {}, "columns_ignored": []}
    warnings, dropped = report["warnings"], report["dropped"]

    def warn(code, message):
        warnings.append({"code": code, "message": message})

    def drop(reason, indices):
        if len(indices):
            entry = dropped.setdefault(reason, {"count": 0, "first_rows": []})
            entry["count"] += len(indices)
            seen = entry["first_rows"]
            seen += [r for r in dict.fromkeys(rows.get(i, i) for i in indices) if r not in seen]
            del seen[8:]

    # ---- columns
    by_name = {_normal(_split_unit(c)[0]): c for c in frame.columns}
    pick = lambda names: next((by_name[n] for n in by_name if n in names), None)
    time_col, date_col, well_col = pick(TIME_NAMES), pick(DATE_NAMES), pick(WELL_NAMES)
    vector_col, value_col, sigma_col, weight_col = pick(VECTOR_NAMES), pick(VALUE_NAMES), pick(SIGMA_NAMES), pick(WEIGHT_NAMES)
    if time_col is None and date_col is None:
        _fail(name, "missing_column", "no time column found", found=list(frame.columns),
              hint="a column named 'time' or 'days' (days since the start) or 'date'")
    if layout == "auto":
        layout = "long" if vector_col and value_col else "by_well" if well_col else "wide"
    report["layout"] = layout
    special = {time_col, date_col, well_col, vector_col, value_col, sigma_col, weight_col}

    # ---- time in days
    if time_col is not None:
        days = pd.to_numeric(frame[time_col], errors="coerce")
        bad = frame.index[days.isna()]
        drop("time is not a number", bad)
        report["time"] = f"column {time_col!r}, days"
    else:
        dates = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=dayfirst)
        drop("date cannot be read", frame.index[dates.isna()])
        origin = pd.Timestamp(start_date) if start_date else dates.min()
        if pd.isna(origin):
            _fail(name, "bad_value", f"no valid date in column {date_col!r}")
        days = (dates - origin).dt.total_seconds() / 86400.0
        report["time"] = f"column {date_col!r}, days since {origin.date()}" + ("" if start_date else " (the first date)")
        report["start_date"] = str(origin.date())

    # ---- model information
    known_wells, kinds = set(), {}
    if model is not None:
        for n, _ in enumerate(model.producers, 1):
            known_wells.add(f"P{n}"), kinds.update({f"P{n}": "producer"})
        for n, _ in enumerate(model.injectors, 1):
            known_wells.add(f"I{n}"), kinds.update({f"I{n}": "injector"})
        dt = dt if dt is not None else model.dt
        t_end = t_end if t_end is not None else model.t_end
    wells = dict(wells or {})

    def kusim_well(raw):
        raw = str(raw).strip()
        if raw in wells:
            return wells[raw]
        if re.fullmatch(r"[PI]\d+", raw):
            return raw
        for key, value in wells.items():
            if key.lower() == raw.lower():
                return value
        return None

    # ---- records: (frame index, well, quantity, value, sigma, weight)
    records: List[tuple] = []
    per_row = lambda i: (float(frame.at[i, sigma_col]) if sigma_col and pd.notna(frame.at[i, sigma_col]) else None,
                         float(frame.at[i, weight_col]) if weight_col and pd.notna(frame.at[i, weight_col]) else None)
    quantity_units = {}
    if layout == "by_well":
        if well_col is None:
            _fail(name, "missing_column", "layout by_well needs a well column", found=list(frame.columns), hint="'Well'")
        quantity_columns = {}
        for column in frame.columns:
            if column in special:
                continue
            quantity = _quantity_of(_split_unit(column)[0])
            if quantity is None:
                report["columns_ignored"].append(column)
            else:
                quantity_columns[column] = quantity
        if not quantity_columns:
            _fail(name, "missing_column", "no column with a known quantity (oil rate, water rate, BHP, water cut, "
                  "water injection)", found=list(frame.columns))
        for i in frame.index:
            well = frame.at[i, well_col]
            if pd.isna(well):
                continue
            s, w = per_row(i)
            for column, quantity in quantity_columns.items():
                records.append((i, str(well).strip(), quantity, frame.at[i, column], s, w, column))
    elif layout == "wide":
        columns = {}
        for column in frame.columns:
            if column in special:
                continue
            parsed = _quantity_and_well(_split_unit(column)[0])
            if parsed is None:
                report["columns_ignored"].append(column)
            else:
                columns[column] = parsed
        if not columns:
            _fail(name, "missing_column", "no column looks like a well and a quantity ('WBHP:P1', 'P1 BHP', ...)",
                  found=list(frame.columns))
        for i in frame.index:
            for column, (quantity, well) in columns.items():
                records.append((i, well, quantity, frame.at[i, column], None, None, column))
    else:                                                                       # long
        if vector_col is None or value_col is None:
            _fail(name, "missing_column", "layout long needs 'vector' and 'value' columns", found=list(frame.columns))
        unknown = set()
        for i in frame.index:
            text = str(frame.at[i, vector_col]).strip()
            s, w = per_row(i)
            match = ECLIPSE_VECTOR.match(text)
            if match:
                prefix = match.group(1).upper()
                quantity = next(q for q, v in {**PRODUCER_VECTORS, "water_inj": "WWIR"}.items() if v == prefix)
                records.append((i, match.group(2).strip(), quantity, frame.at[i, value_col], s, w, vector_col))
            elif well_col and _quantity_of(_split_unit(text)[0]):
                records.append((i, str(frame.at[i, well_col]).strip(), _quantity_of(_split_unit(text)[0]),
                                frame.at[i, value_col], s, w, vector_col))
            else:
                unknown.add(text)
        if unknown:
            _fail(name, "unknown_vector", f"cannot tell what these entries of {vector_col!r} are: {sorted(unknown)[:8]}",
                  allowed=["WBHP:<well>", "WOPR:<well>", "WWPR:<well>", "WWCT:<well>", "WWIR:<well>"],
                  hint="use ECLIPSE vector names, or a quantity name with a 'well' column")

    # ---- blank and non-numeric cells are skipped (and counted) before anything is checked
    clean = []
    for rec in records:
        i, value = rec[0], rec[3]
        if pd.isna(days.get(i)):
            continue                                                         # the row was counted as having no time
        if value is None or (isinstance(value, float) and np.isnan(value)) or (isinstance(value, str) and not value.strip()):
            drop("blank value", [i])
            continue
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            drop("value is not a number", [i])
            continue
        clean.append((*rec[:3], float(number), *rec[4:]))
    records = clean
    if not records:
        _fail(name, "no_data", "the sheet has no usable measurements", dropped=dropped)

    # ---- wells that the caller leaves out on purpose
    if skip_wells:
        skip = {str(s).strip().lower() for s in skip_wells}
        present = {str(r[1]).strip().lower(): str(r[1]) for r in records}
        missing = sorted(skip - set(present))
        if missing:
            _fail(name, "unknown_well", f"skip_wells names wells that the sheet does not have: {missing}", wells_in_sheet=sorted(present.values()))
        kept = [r for r in records if str(r[1]).strip().lower() not in skip]
        drop("well skipped on request", [r[0] for r in records if str(r[1]).strip().lower() in skip])
        report["skipped_wells"] = sorted(present[s] for s in skip)
        records = kept
        if not records:
            _fail(name, "no_data", "every well of the sheet was skipped", dropped=dropped)

    # ---- wells and vectors
    well_names = sorted({r[1] for r in records})
    mapped = {w: kusim_well(w) for w in well_names}
    shared = {}
    for w, k in mapped.items():
        if k is not None:
            shared.setdefault(k, []).append(w)
    shared = {k: ws for k, ws in shared.items() if len(ws) > 1}
    if shared:
        _fail(name, "duplicate_well_map", f"several wells of the sheet are mapped to the same kusim well: {shared}. That would put the "
              "measurements of different wells under one name. Every well of the sheet needs its own kusim well",
              wells_in_sheet=well_names, shared=shared,
              hint="if the model has fewer wells than the data, the data and the model describe different reservoirs: stop and tell the user, "
                   "or use a model that has these wells (import-deck of the field's deck). To leave a well out on purpose, say so with "
                   "--skip-well NAME and report it")
    unmapped = [w for w, k in mapped.items() if k is None]
    if unmapped:
        _fail(name, "unmapped_well", f"the sheet's wells {unmapped} have no kusim name (P1, P2, ..., I1, ...)",
              wells_in_sheet=well_names, hint="give a map: {'PROD-A': 'P1'} (--well-map PROD-A=P1 or --wells-file from "
              "import-deck). Map each well to its own kusim well; do not map two wells to one. If the model does not have a well of "
              "the data, the model is not the one that produced the data: tell the user, or leave the well out on purpose with --skip-well NAME")
    if model is not None:
        wrong = [f"{w} -> {k}" for w, k in mapped.items() if k not in known_wells]
        if wrong:
            _fail(name, "unknown_well", f"the model does not have the wells {wrong}", allowed=sorted(known_wells),
                  hint="do not map a well to a different kusim well to get past this error: that mixes the data of two wells. The model "
                       "does not have this well, so it is not the model of this field: tell the user, use a model that has the well, or "
                       "leave the well out on purpose with --skip-well NAME and report it")
    report["wells"] = {w: k for w, k in mapped.items()}

    def kind_of(kusim):
        if kusim in kinds:
            return kinds[kusim]
        return "injector" if kusim.startswith("I") else "producer"

    resolved, bad_vector = [], []
    for i, well, quantity, value, s, w, column in records:
        kusim = mapped[well]
        table = INJECTOR_VECTORS if kind_of(kusim) == "injector" else PRODUCER_VECTORS
        if quantity == "liquid_rate":
            vector = "liquid"                                  # not a kusim vector: only for controls
        elif quantity not in table:
            bad_vector.append(f"{well}: {quantity}")
            continue
        else:
            vector = table[quantity]
        resolved.append((i, kusim, quantity, vector, value, s, w, column))
    if bad_vector:
        _fail(name, "bad_vector_for_well", f"quantities that the well type does not have: {sorted(set(bad_vector))[:8]}",
              hint="an injector has BHP and water injection; a producer has BHP, oil rate, water rate and water cut")

    # ---- values, units
    unit_options = {"pressure": pressure_unit, "rate": rate_unit, "water_cut": water_cut_unit}
    rows_out = []
    for i, kusim, quantity, vector, number, s, w, column in resolved:
        kind = UNIT_OF.get(quantity, "rate")
        header_unit = _split_unit(column)[1] if layout != "long" else None
        unit = header_unit or unit_options[kind]
        factor = _convert(kind, unit, f"{name}:{column}")
        quantity_units[(column, kind)] = unit
        rows_out.append((i, kusim, quantity, vector, number * factor, s, w, float(days[i])))
    report["units"] = {f"{c}": u for (c, _), u in quantity_units.items()}
    frame_out = pd.DataFrame(rows_out, columns=["row", "well", "quantity", "vector", "value", "sigma", "weight", "time"])
    if frame_out.empty:
        _fail(name, "no_data", "the sheet has no usable measurements", dropped=dropped)

    trim = frame_out[(frame_out.quantity == "water_cut") & (frame_out.value < 0)]
    if len(trim):                                                         # noise around zero: a water cut cannot be negative
        report["trimmed"] = {"water cut below 0 set to 0": {"count": len(trim), "first_rows": sorted(set(trim["row"].map(
            lambda i: rows.get(i, i))))[:8]}}
        frame_out.loc[trim.index, "value"] = 0.0
        warn("water_cut_trimmed", f"{len(trim)} negative water cut values were set to 0")
    negative = frame_out[(frame_out.value < 0)]
    drop("negative value", negative["row"])
    frame_out = frame_out.drop(negative.index)
    cut = frame_out[(frame_out.quantity == "water_cut") & (frame_out.value > 1.0 + 1e-9)]
    if len(cut):
        _fail(name, "bad_value", f"water cut above 1 (rows {[rows.get(i, i) for i in cut['row'][:5]]}): is it in percent?",
              hint="use water_cut_unit='percent' (--water-cut-unit percent) or '(%)' in the header")
    before = frame_out[frame_out.time <= 0]
    drop("time at or before day 0", before["row"])
    frame_out = frame_out.drop(before.index)
    if t_end is not None:
        late = frame_out[frame_out.time > t_end + 1e-9]
        drop(f"after the end time ({t_end:g} days)", late["row"])
        frame_out = frame_out.drop(late.index)
    if vectors is not None:
        wanted = {v.upper() for v in vectors}
        unknown_vectors = wanted - set(VECTOR_PREFIXES)
        if unknown_vectors:
            _fail("vectors", "bad_value", f"unknown vectors {sorted(unknown_vectors)}", allowed=list(VECTOR_PREFIXES))
    else:
        wanted = set(VECTOR_PREFIXES)
    if frame_out.empty:
        _fail(name, "no_data", "no measurement is left after the time window", dropped=dropped,
              hint="check start_date, the end time and that the time column is in days or dates")

    # ---- alignment to the time step
    every = every if every is not None else dt
    if every is not None:
        if dt is not None and abs(every / dt - round(every / dt)) > 1e-9:
            _fail("every", "bad_value", f"every = {every:g} is not a multiple of dt = {dt:g}")
        frame_out["bin"] = np.ceil(frame_out.time / every - 1e-9).astype(int)
        report["aligned"] = f"intervals of {every:g} days; a value at day T summarises the interval (T - {every:g}, T]"
    else:
        warn("times_not_aligned", "no dt or model was given: the times are kept as they are, and misfit needs multiples of "
             "the model's time step")

    # ---- observations
    default_sigma = dict(DEFAULT_SIGMA)
    for key, value in (sigma or {}).items():
        default_sigma[_prefix_of(key, "sigma")] = value
    given = {_prefix_of(k, "sigma") for k in (sigma or {})}
    observed = frame_out[frame_out.vector.isin(wanted)]
    out = []
    group_keys = ["well", "vector"] + (["bin"] if every is not None else ["time"])
    for keys, group in observed.groupby(group_keys):
        well, vector = keys[0], keys[1]
        time = keys[2] * every if every is not None else keys[2]
        group = group.sort_values("time")
        value = group.value.mean() if align == "mean" else group.value.iloc[-1]
        sigma_value = group.sigma.dropna().mean() if group.sigma.notna().any() else None
        weight = group.weight.dropna().mean() if group.weight.notna().any() else 1.0
        out.append({"time": float(time), "vector": f"{vector}:{well}", "value": float(value), "sigma": sigma_value,
                    "weight": float(weight)})
    table = pd.DataFrame(out)
    if table.empty:
        _fail(name, "no_data", "no observation is left after choosing the vectors", vectors=sorted(wanted))
    for prefix in sorted({v.split(":")[0] for v in table.vector}):
        mask = (table.vector.str.split(":").str[0] == prefix)
        missing = mask & table.sigma.isna()
        if not missing.any():
            continue
        kind, size = _parse_sigma(default_sigma[prefix], f"sigma {prefix}")
        if kind == "abs":
            table.loc[missing, "sigma"] = size
        else:
            floor = max(0.01 * table.loc[mask, "value"].abs().max(), 1e-9)
            table.loc[missing, "sigma"] = np.maximum(size * table.loc[missing, "value"].abs(), floor)
        report["defaults_used"][prefix] = ("given in the options" if prefix in given else "default") + \
            f": {default_sigma[prefix]}" + (" (a percentage of the value, at least 1 % of the largest value)" if kind == "rel" else "")
        if prefix not in given:
            warn("default_sigma", f"{prefix}: no sigma given, using {default_sigma[prefix]}"
                 + (" of the value" if kind == "rel" else "") + " (the standard deviation of the measurement error: set it with "
                 "sigma / --sigma)")
    observations = Observations(table)
    report["observations"] = {"n": len(table), "vectors": {v: int(c) for v, c in table.vector.value_counts().sort_index().items()},
                              "time_range": [float(table.time.min()), float(table.time.max())]}

    # ---- controls from measured rates
    events = []
    if controls:
        if every is None:
            _fail("controls", "controls_need_dt", "controls from measured rates need dt (or a model): the rates must be "
                  "grouped into time steps", hint="give dt or model")
        events = _controls(frame_out, every, align, producer_control, kinds, t_end, warn, name)
    report["controls"] = {"n_events": len(events), "producer_control": producer_control} if controls else None
    return HistoryImport(observations, Schedule(events), report)


def _controls(frame, every, align, producer_control, kinds, t_end, warn, name):
    """Events that impose the measured rates: each interval's rate acts from the start of the interval."""
    events = []
    rate_quantities = {"oil_rate", "water_rate", "liquid_rate", "water_inj"}
    data = frame[frame.quantity.isin(rate_quantities)]
    for well, rows in data.groupby("well"):
        injector = kinds.get(well, "injector" if well.startswith("I") else "producer") == "injector"
        series = {}
        for quantity, part in rows.groupby("quantity"):
            grouped = part.groupby("bin").value
            series[quantity] = grouped.mean() if align == "mean" else grouped.last()
        if injector:
            source, mode = series.get("water_inj", series.get("water_rate")), "water_rate"
        elif producer_control == "oil_rate":
            source, mode = series.get("oil_rate"), "oil_rate"
        elif "liquid_rate" in series:
            source, mode = series["liquid_rate"], "liquid_rate"
        elif "oil_rate" in series and "water_rate" in series:
            source, mode = series["oil_rate"].add(series["water_rate"], fill_value=0.0), "liquid_rate"
        else:
            source, mode = None, producer_control
        if source is None or source.empty:
            warn("no_rates_for_controls", f"well {well}: no rates to make controls from (oil rate"
                 + ("" if producer_control == "oil_rate" else ", liquid rate or oil and water rates") + ")")
            continue
        last = None
        for bin_index, value in source.sort_index().items():
            start = (bin_index - 1) * every
            if t_end is not None and start >= t_end - 1e-9:
                continue
            if last is None or abs(value - last) > 1e-9 * max(abs(value), 1.0):
                events.append(ControlEvent(float(start), well, mode, float(value)))
                last = value
        bins = sorted(source.index)
        if bins and bins[-1] - bins[0] + 1 != len(bins):
            warn("gaps_in_rates", f"well {well}: rates are not given for every {every:g}-day interval; the last rate stays in force until the next one")
    return events


def write_history_template(path):
    """Write a small example workbook: a sheet in the by_well layout (with units in the headers) and a sheet that explains it."""
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "history"
    sheet.append(["Date", "Well", "Oil rate (STB/d)", "Water rate (STB/d)", "BHP (psi)", "Water injection (STB/d)"])
    rng = np.random.default_rng(1)
    for month in range(1, 13):
        date = pd.Timestamp(2020, month, 1).to_pydatetime()
        for well, base in (("P1", 2950.0), ("P2", 3100.0)):
            water = round(float(month ** 2 * 0.4 + rng.normal(0, 1)), 1)
            sheet.append([date, well, round(base - month * 5 + float(rng.normal(0, 20)), 1), max(water, 0.0),
                          round(1500.0 + float(rng.normal(0, 15)), 1), None])
        sheet.append([date, "I1", None, None, round(3000.0 + month * 8 + float(rng.normal(0, 15)), 1), 6000.0])
    notes = book.create_sheet("README")
    for line in (
        "An example of a history workbook for kusim (these numbers are made up).",
        "One row per date and well; one column per quantity. Put the unit in the header: BHP (bar), Oil rate (m3/d), Water cut (%).",
        "Recognised quantities: oil rate, water rate, liquid rate, BHP (or pressure), water cut, water injection.",
        "Layouts: 'by_well' (this sheet), 'wide' (one column per well and quantity: WBHP:P1, P1 BHP) and 'long' (time, vector, value, sigma).",
        "Time: a 'Date' column (days are counted from the first date, or from --start-date) or a 'Time' column of days.",
        "Blank cells are skipped (and counted). Names of wells are mapped to kusim's P1, P2, I1 ... by --well-map or --wells-file.",
        "Command:  kusim import-history history.xlsx -o observations.json --model model.json --sigma WBHP=25 --controls-out schedule.json",
    ):
        notes.append([line])
    notes.column_dimensions["A"].width = 120
    book.save(path)
