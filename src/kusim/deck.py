"""Read an ECLIPSE-style input deck (``.DATA``) into a kusim model and schedule.

    imported = read_deck("CASE.DATA")
    imported.model          # a ReservoirModel
    imported.schedule       # a Schedule (well-control events after day 0)
    imported.wells          # {"PROD1": "P1", "INJ1": "I1"}: deck well names and kusim's names
    imported.warnings       # things that were mapped with a remark (each: code, message, where)

Only what kusim can represent is read. The reader is strict: a keyword or a value that kusim cannot honour stops the
import with a located ``SpecError`` (file and line, a code and a hint) instead of being skipped, because a skipped keyword
changes the answer silently. ``skip_unknown=True`` turns those errors into warnings (the data of an unknown keyword is
skipped up to the next keyword that this module knows, so use it with care). Keywords that have no effect on the result
(``TITLE``, ``RPTSCHED``, the ``SUMMARY`` section, ...) are accepted and listed in ``imported.ignored``.

Supported (oil-water, FIELD units, one region, Cartesian grid):

    RUNSPEC  DIMENS, FIELD, OIL, WATER, START
    GRID     DX DY DZ (or DXV DYV DZV), TOPS, PORO, PERMX, PERMY, PERMZ, and EQUALS, MULTIPLY, ADD, COPY on them
             (NTG, ACTNUM only when they are all 1)
    PROPS    PVDO or PVCDO, PVTW, ROCK, DENSITY, SWOF or SWFN + SOF2
    SOLUTION EQUIL (datum depth and pressure, water-oil contact; zero capillary pressure there), or uniform PRESSURE
             and SWAT
    SCHEDULE WELSPECS, COMPDAT, WCONPROD, WCONHIST, WCONINJE, WELTARG, WELOPEN, TSTEP, DATES, END
    anything   INCLUDE

How it maps: ``EQUIL`` becomes ``initial.datum_depth``, ``datum_pressure`` and ``woc_depth``; the depth frame of the deck
(``TOPS`` and so on) is kept. kusim names wells ``P1, P2, ...`` and ``I1, ...`` (producers and injectors each in the order
of ``WELSPECS``); ``imported.wells`` gives the deck names. Controls given before the first time step are the controls of
the model, later ones are schedule events. The time step ``dt`` is the greatest common divisor of the ``TSTEP``/``DATES``
intervals unless given; every report time and event time must be a multiple of it.
"""

from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .fluids import PVTTable, RockFluidTable
from .grid import Grid
from .model import Equilibrium, ReservoirModel, UniformStart
from .schedule import ControlEvent, Schedule
from .spec import SpecError
from .wells import Injector, Producer

WATER_DENSITY = 62.4            # lb/ft3: gamma = density / 62.4 (the 0.433 psi/ft of the water gradient)
DEFAULT_WELL_RADIUS = 0.5       # ft: ECLIPSE's default well diameter is 1 ft
SECTIONS = {"RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SCHEDULE"}
ARRAYS = {"PORO", "PERMX", "PERMY", "PERMZ", "DX", "DY", "DZ", "TOPS", "NTG", "ACTNUM", "PRESSURE", "SWAT"}
REGION_ARRAYS = {"SATNUM", "PVTNUM", "EQLNUM", "FIPNUM", "ENDNUM"}          # accepted when they are all 1
VECTORS = {"DXV", "DYV", "DZV"}
TABLES = {"PVDO": 3, "SWOF": 4, "SWFN": 3, "SOF2": 2}
SINGLE = {"DIMENS", "START", "DENSITY", "ROCK", "PVTW", "PVCDO", "EQUIL", "TSTEP"}
RECORDS = {"WELSPECS", "COMPDAT", "WCONPROD", "WCONHIST", "WCONINJE", "WELTARG", "WELOPEN", "EQUALS", "MULTIPLY", "ADD",
           "COPY", "DATES"}
FLAGS = {"RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SCHEDULE", "OIL", "WATER", "FIELD", "UNIFOUT",
         "UNIFIN", "NOECHO", "ECHO", "NOSIM", "INIT", "SKIPREST", "FMTOUT", "FMTIN", "MULTOUT", "END", "SUMMARY"}
IGNORED_SINGLE = {"TABDIMS", "EQLDIMS", "WELLDIMS", "GRIDOPTS", "NSTACK", "MESSAGES", "RPTSCHED", "RPTSOL", "RPTRST",
                  "RPTGRID", "RPTPROPS", "RPTREGS", "GRIDFILE", "OUTSOL", "REGDIMS", "AQUDIMS", "NUPCOL", "MAPAXES",
                  "GRIDUNIT", "VFPIDIMS", "SUMTHIN", "NOWARNEP", "FAULTDIM", "TITLE"}
IGNORED_RECORDS = {"GRUPTREE"}
UNSUPPORTED = {
    "METRIC": ("unsupported_units", "kusim works in oilfield units: convert the deck to FIELD units"),
    "LAB": ("unsupported_units", "kusim works in oilfield units: convert the deck to FIELD units"),
    "GAS": ("unsupported_feature", "kusim is a two-phase oil-water simulator"),
    "DISGAS": ("unsupported_feature", "kusim has no dissolved gas: use dead oil (PVDO or PVCDO)"),
    "VAPOIL": ("unsupported_feature", "kusim is a two-phase oil-water simulator"),
    "PVTO": ("unsupported_feature", "live oil is not supported: use PVDO (Bo and viscosity against pressure) or PVCDO"),
    "PVTG": ("unsupported_feature", "kusim is a two-phase oil-water simulator"),
    "PVDG": ("unsupported_feature", "kusim is a two-phase oil-water simulator"),
    "SGOF": ("unsupported_feature", "kusim is a two-phase oil-water simulator"),
    "SGFN": ("unsupported_feature", "kusim is a two-phase oil-water simulator"),
    "SOF3": ("unsupported_feature", "kusim is a two-phase oil-water simulator: use SWOF"),
    "COORD": ("unsupported_geometry", "corner-point grids are not supported: kusim has a Cartesian grid (DX, DY, DZ, TOPS)"),
    "ZCORN": ("unsupported_geometry", "corner-point grids are not supported: kusim has a Cartesian grid (DX, DY, DZ, TOPS)"),
    "SPECGRID": ("unsupported_geometry", "use DIMENS and DX, DY, DZ, TOPS"),
    "DEPTHZ": ("unsupported_geometry", "use TOPS (the depth of the top of the first layer)"),
    "BOX": ("unsupported_feature", "BOX/ENDBOX are not supported: give the box to EQUALS, MULTIPLY, ADD or COPY"),
    "ENDBOX": ("unsupported_feature", "BOX/ENDBOX are not supported: give the box to EQUALS, MULTIPLY, ADD or COPY"),
    "MULTX": ("unsupported_feature", "transmissibility multipliers are not supported: edit the permeability instead"),
    "MULTY": ("unsupported_feature", "transmissibility multipliers are not supported: edit the permeability instead"),
    "MULTZ": ("unsupported_feature", "transmissibility multipliers are not supported: edit the permeability instead"),
    "MULTPV": ("unsupported_feature", "pore volume multipliers are not supported: edit the porosity instead"),
    "SWL": ("unsupported_feature", "end-point scaling is not supported: put the end points into the SWOF table"),
    "SWCR": ("unsupported_feature", "end-point scaling is not supported: put the end points into the SWOF table"),
    "SOWCR": ("unsupported_feature", "end-point scaling is not supported: put the end points into the SWOF table"),
    "ENDSCALE": ("unsupported_feature", "end-point scaling is not supported"),
    "SWATINIT": ("unsupported_feature", "SWATINIT is not supported: use EQUIL or a uniform SWAT"),
    "RSVD": ("unsupported_feature", "kusim has no dissolved gas"),
    "AQUFETP": ("unsupported_feature", "aquifers are not supported"),
    "AQUCT": ("unsupported_feature", "aquifers are not supported"),
    "AQUANCON": ("unsupported_feature", "aquifers are not supported"),
    "THPRES": ("unsupported_feature", "threshold pressures are not supported"),
    "WPIMULT": ("unsupported_feature", "well index multipliers are not supported: change the skin or the radius"),
    "WELPI": ("unsupported_feature", "kusim computes the well index itself (Peaceman)"),
    "GCONPROD": ("unsupported_feature", "group controls are not supported: control the wells"),
    "GCONINJE": ("unsupported_feature", "group controls are not supported: control the wells"),
    "VFPPROD": ("unsupported_feature", "vertical lift tables are not supported"),
    "WTEST": ("unsupported_feature", "well testing is not supported"),
    "HYSTER": ("unsupported_feature", "hysteresis is not supported"),
}
KNOWN = (SECTIONS | ARRAYS | REGION_ARRAYS | VECTORS | set(TABLES) | SINGLE | RECORDS | FLAGS | IGNORED_SINGLE
         | IGNORED_RECORDS | set(UNSUPPORTED) | {"INCLUDE"})
MONTHS = {m: n for n, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
MONTHS["JLY"] = 7


@dataclass
class DeckImport:
    """What ``read_deck`` returns."""

    model: ReservoirModel
    schedule: Schedule
    wells: Dict[str, str]                                   # deck well name -> kusim name
    warnings: List[dict] = field(default_factory=list)      # {"code", "message", "where"}
    ignored: List[dict] = field(default_factory=list)       # {"keyword", "where", "reason"}
    mapping: List[str] = field(default_factory=list)        # one line per piece of the deck: where it went in the model


@dataclass
class _Token:
    text: str
    quoted: bool
    file: str
    line: int
    first: bool                                             # first token on its line (where keywords start)

    @property
    def where(self):
        return f"{self.file}:{self.line}"


def _fail(where, code, message, **details):
    raise SpecError(message, where, code, **details)


# ---------------------------------------------------------------------- tokens
def _tokenise(text, file):
    tokens = []
    for number, raw in enumerate(text.splitlines(), start=1):
        i, n, first = 0, len(raw), True
        while i < n:
            c = raw[i]
            if c.isspace():
                i += 1
            elif raw.startswith("--", i):
                break
            elif c == "/":
                tokens.append(_Token("/", False, file, number, first))
                i, first = i + 1, False
            elif c in "'\"":
                end = raw.find(c, i + 1)
                if end < 0:
                    _fail(f"{file}:{number}", "syntax", "a quoted string is not closed")
                tokens.append(_Token(raw[i + 1:end], True, file, number, first))
                i, first = end + 1, False
            else:
                j = i
                while j < n and not raw[j].isspace() and raw[j] != "/":
                    if raw[j] in "'\"":                       # n*'text'
                        end = raw.find(raw[j], j + 1)
                        j = n if end < 0 else end
                    j += 1
                tokens.append(_Token(raw[i:j], False, file, number, first))
                i, first = j, False
    return tokens


def _value(token):
    """A number, ``None`` for a defaulted item (``1*``), or the text."""
    if token.quoted:
        return token.text
    text = token.text
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None if text == "*" else text


def _expand(tokens):
    """The values of a list of tokens: ``n*v`` repeated, ``n*`` as defaults."""
    out = []
    for token in tokens:
        text = token.text
        if not token.quoted and "*" in text and text != "*":
            count, _, rest = text.partition("*")
            try:
                n = int(count)
            except ValueError:
                _fail(token.where, "bad_value", f"{text!r} is not a number or a repeat like 3*0.2")
            value = None if rest == "" else _value(_Token(rest.strip("'\""), rest[:1] in "'\"", token.file, token.line, False))
            out.extend([value] * n)
        else:
            out.append(_value(token))
    return out


class _Record:
    """The items of one record (a line of WELSPECS, the data of PVTW, ...)."""

    def __init__(self, keyword, tokens, where):
        self.keyword, self.where = keyword, where
        self.items = _expand(tokens)

    def get(self, i):
        return self.items[i] if i < len(self.items) else None

    def number(self, i, name, default=None):
        value = self.get(i)
        if value is None:
            if default is None:
                _fail(self.where, "missing_value", f"{self.keyword}: item {i + 1} ({name}) is needed")
            return default
        if isinstance(value, str):
            _fail(self.where, "bad_value", f"{self.keyword}: item {i + 1} ({name}) must be a number, got {value!r}")
        return float(value)

    def integer(self, i, name, default=None):
        return int(round(self.number(i, name, default)))

    def text(self, i, default=None):
        value = self.get(i)
        if value is None:
            return default
        return value if isinstance(value, str) else repr(value)

    def upper(self, i, default=None):
        value = self.text(i, default)
        return value.upper() if isinstance(value, str) else value


class _Stream:
    def __init__(self, tokens):
        self.tokens, self.i, self.includes = tokens, 0, 0

    def done(self):
        return self.i >= len(self.tokens)

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self):
        token = self.tokens[self.i]
        self.i += 1
        return token

    def until_slash(self):
        out = []
        while not self.done():
            token = self.next()
            if token.text == "/" and not token.quoted:
                return out
            out.append(token)
        return out

    def records(self):
        out = []
        while not self.done():
            record = self.until_slash()
            if not record:
                break
            out.append(record)
        return out

    def skip_to_keyword(self):
        """After a keyword that is not understood: skip to the next token that starts a known keyword."""
        while not self.done():
            token = self.peek()
            if token.first and not token.quoted and token.text.upper()[:8] in KNOWN:
                return
            self.i += 1


def _is_number(token):
    return token is not None and not token.quoted and isinstance(_value(token), float)


# ---------------------------------------------------------------------- the reader
class _Reader:
    def __init__(self, path, dt, skip_unknown, pvt_pressures):
        self.path = Path(path)
        self.dt_option, self.skip_unknown, self.pvt_pressures = dt, skip_unknown, pvt_pressures
        self.warnings, self.ignored, self.mapping = [], [], []
        self.dims = None
        self.units = None
        self.arrays: Dict[str, np.ndarray] = {}
        self.array_where: Dict[str, str] = {}
        self.vectors: Dict[str, list] = {}
        self.regions = {}
        self.tables: Dict[str, list] = {}
        self.single: Dict[str, _Record] = {}
        self.start = None
        self.now = 0.0
        self.report_times: List[float] = []
        self.wells: Dict[str, dict] = {}
        self.ended = False

    # -- helpers
    def warn(self, where, code, message):
        self.warnings.append({"code": code, "message": message, "where": where})

    def ignore(self, keyword, where, reason):
        self.ignored.append({"keyword": keyword, "where": where, "reason": reason})

    def need_dims(self, where, keyword):
        if self.dims is None:
            _fail(where, "missing_keyword", f"{keyword} comes before DIMENS", hint="put DIMENS in RUNSPEC")
        return self.dims

    # -- main loop
    def read(self):
        path = self.path
        try:
            text = path.read_text(encoding="utf8", errors="replace")
        except OSError:
            _fail(str(path), "file_not_found", f"cannot read {path}")
        stream = _Stream(_tokenise(text, str(path.name)))
        self.stream = stream
        while not stream.done() and not self.ended:
            token = stream.next()
            name = token.text.upper()[:8]
            if token.quoted or not re.match(r"^[A-Z][A-Z0-9_-]*$", token.text.upper()):
                _fail(token.where, "syntax", f"expected a keyword, found {token.text!r}",
                      hint="a data item that is left over from the keyword before, or a missing '/'")
            self.handle(token, name)

    def handle(self, token, name):
        s = self.stream
        if name == "INCLUDE":
            record = s.until_slash()
            if not record:
                _fail(token.where, "missing_value", "INCLUDE needs a file name")
            target = (self.path.parent / record[0].text).resolve()
            s.includes += 1
            if s.includes > 50:
                _fail(token.where, "syntax", "too many INCLUDE files (a loop?)")
            if not target.exists():
                _fail(token.where, "file_not_found", f"INCLUDE file {record[0].text!r} not found next to {self.path.name}")
            new = _tokenise(target.read_text(encoding="utf8", errors="replace"), target.name)
            s.tokens[s.i:s.i] = new
        elif name == "SUMMARY":
            self.ignore("SUMMARY", token.where, "output requests (the section is skipped)")
            while not s.done():
                t = s.peek()
                if t.first and t.text.upper() in ("SCHEDULE", "END", "SOLUTION", "RUNSPEC", "GRID", "PROPS"):
                    break
                s.i += 1
        elif name in ("GAS", "DISGAS", "VAPOIL") and name in UNSUPPORTED:
            self.unsupported(token, name)
        elif name == "FIELD":
            self.units = "FIELD"
        elif name == "END":
            self.ended = True
        elif name in FLAGS:
            if name not in SECTIONS | {"OIL", "WATER"}:
                self.ignore(name, token.where, "no effect on the model")
        elif name == "TITLE":
            first = s.peek()
            while not s.done() and s.peek().line == first.line:
                s.i += 1
            self.ignore("TITLE", token.where, "a label")
        elif name == "DIMENS":
            record = _Record(name, s.until_slash(), token.where)
            self.dims = tuple(record.integer(i, "n" + "xyz"[i]) for i in range(3))
        elif name == "START":
            self.start = self.parse_date(_Record(name, s.until_slash(), token.where))
        elif name in ARRAYS or name in REGION_ARRAYS:
            self.read_array(token, name)
        elif name in VECTORS:
            self.vectors[name] = [v for v in _expand(s.until_slash())]
        elif name in TABLES:
            self.tables[name] = self.read_table(token, name)
        elif name in ("DENSITY", "ROCK", "PVTW", "PVCDO", "EQUIL"):
            tokens = s.until_slash()
            self.refuse_regions(token, name)
            self.single[name] = _Record(name, tokens, token.where)
        elif name == "TSTEP":
            for v in _expand(s.until_slash()):
                if not isinstance(v, float) or v <= 0:
                    _fail(token.where, "bad_value", f"TSTEP needs positive numbers, got {v!r}")
                self.advance(self.now + v)
        elif name == "DATES":
            for tokens in s.records():
                if self.start is None:
                    _fail(token.where, "missing_keyword", "DATES needs START (the date of day 0)",
                          hint="put START 1 JAN 2020 / in RUNSPEC")
                self.advance((self.parse_date(_Record(name, tokens, token.where)) - self.start).total_seconds() / 86400.0)
        elif name in ("EQUALS", "MULTIPLY", "ADD", "COPY"):
            for tokens in s.records():
                self.operate(name, _Record(name, tokens, token.where))
        elif name in ("WELSPECS", "COMPDAT", "WCONPROD", "WCONHIST", "WCONINJE", "WELTARG", "WELOPEN"):
            for tokens in s.records():
                getattr(self, "do_" + name.lower())(_Record(name, tokens, tokens[0].where))
        elif name in IGNORED_SINGLE:
            s.until_slash()
            self.ignore(name, token.where, "no effect on the model")
        elif name in IGNORED_RECORDS:
            s.records()
            self.ignore(name, token.where, "no effect on the model")
        elif name in UNSUPPORTED:
            self.unsupported(token, name)
        else:
            self.unsupported(token, name, unknown=True)

    def unsupported(self, token, name, unknown=False):
        code, hint = ("unsupported_keyword", "see the list of supported keywords in the documentation (docs/eclipse-deck.md)") \
            if unknown else UNSUPPORTED[name]
        name = token.text.upper() if unknown else name
        message = f"{name} is not supported" + (" by kusim" if unknown else "")
        if not self.skip_unknown:
            _fail(token.where, code, message, keyword=name, hint=hint,
                  skip_with="use skip_unknown (--skip-unknown) to import without it, if you are sure it does not matter")
        self.warn(token.where, code, f"{message} and was skipped ({hint})")
        self.ignore(name, token.where, "skipped (not supported)")
        self.stream.skip_to_keyword()

    def refuse_regions(self, token, name):
        if _is_number(self.stream.peek()):
            _fail(token.where, "unsupported_feature", f"{name} has more than one region: kusim has one set of properties",
                  hint="reduce the deck to one SATNUM/PVTNUM/EQLNUM region")

    # -- grid arrays
    def read_array(self, token, name):
        nx, ny, nz = self.need_dims(token.where, name)
        n = nx * ny * nz
        values = _expand(self.stream.until_slash())
        self.refuse_regions(token, name)
        if any(not isinstance(v, float) for v in values):
            bad = next(v for v in values if not isinstance(v, float))
            _fail(token.where, "bad_value", f"{name} must hold numbers, found {bad!r}")
        allowed = {n, nx * ny} if name == "TOPS" else {n}
        if len(values) not in allowed:
            _fail(token.where, "bad_value", f"{name} has {len(values)} values; the grid {nx} x {ny} x {nz} needs "
                  + " or ".join(str(a) for a in sorted(allowed)))
        array = np.full(n, np.nan)
        array[:len(values)] = values
        self.arrays[name] = array
        self.array_where[name] = token.where

    def box(self, record, first, token_name):
        nx, ny, nz = self.dims
        limits = []
        for i, top in zip(range(first, first + 6), (nx, nx, ny, ny, nz, nz)):
            v = record.get(i)
            limits.append(None if v is None else int(v))
        i1, i2, j1, j2, k1, k2 = (d if v is None else v for v, d in zip(limits, (1, nx, 1, ny, 1, nz)))
        if not (1 <= i1 <= i2 <= nx and 1 <= j1 <= j2 <= ny and 1 <= k1 <= k2 <= nz):
            _fail(record.where, "bad_value", f"{token_name}: the box {i1}-{i2}, {j1}-{j2}, {k1}-{k2} is outside the "
                  f"{nx} x {ny} x {nz} grid")
        mask = np.zeros((nz, ny, nx), bool)
        mask[k1 - 1:k2, j1 - 1:j2, i1 - 1:i2] = True
        return mask.reshape(-1)

    def operate(self, name, record):
        nx, ny, nz = self.need_dims(record.where, name)
        target = record.upper(1 if name == "COPY" else 0)
        if target not in ARRAYS:
            _fail(record.where, "unsupported_keyword", f"{name} on {target!r} is not supported",
                  allowed=sorted(ARRAYS), hint="only the grid arrays that kusim reads can be edited")
        if name == "COPY":
            source = record.upper(0)
            if source not in self.arrays:
                _fail(record.where, "missing_keyword", f"COPY: {source} is not defined yet")
            mask = self.box(record, 2, name)
            self.arrays.setdefault(target, np.full(len(self.arrays[source]), np.nan))
            self.arrays[target][mask] = self.arrays[source][mask]
            return
        value = record.number(1, "value")
        mask = self.box(record, 2, name)
        array = self.arrays.setdefault(target, np.full(nx * ny * nz, np.nan))
        if name == "EQUALS":
            array[mask] = value
        elif target not in self.arrays or np.isnan(array[mask]).any():
            _fail(record.where, "missing_keyword", f"{name}: {target} is not defined everywhere in that box yet")
        elif name == "MULTIPLY":
            array[mask] *= value
        else:
            array[mask] += value

    # -- tables
    def read_table(self, token, name):
        values = _expand(self.stream.until_slash())
        self.refuse_regions(token, name)
        columns = TABLES[name]
        if not values or len(values) % columns or any(not isinstance(v, float) for v in values):
            _fail(token.where, "bad_value", f"{name} needs rows of {columns} numbers (no defaulted items)")
        return np.array(values, float).reshape(-1, columns)

    # -- time
    def parse_date(self, record):
        day = record.integer(0, "day")
        month = record.upper(1)
        if month not in MONTHS and not (isinstance(month, str) and month[:3] in MONTHS):
            _fail(record.where, "bad_value", f"{record.keyword}: {record.get(1)!r} is not a month (JAN ... DEC)")
        month = MONTHS.get(month, MONTHS.get(month[:3]))
        year = record.integer(2, "year")
        hours = minutes = seconds = 0
        clock = record.text(3)
        if clock:
            try:
                hours, minutes, seconds = (int(float(p)) for p in (clock.split(":") + ["0", "0"])[:3])
            except ValueError:
                _fail(record.where, "bad_value", f"{record.keyword}: {clock!r} is not a time (hh:mm:ss)")
        try:
            return datetime(year, month, day, hours, minutes, seconds)
        except ValueError as error:
            _fail(record.where, "bad_value", f"{record.keyword}: {error}")

    def advance(self, time):
        if time <= self.now + 1e-9:
            _fail(self.stream.tokens[self.stream.i - 1].where, "bad_value",
                  f"the time must increase: day {time:g} does not follow day {self.now:g}")
        self.now = time
        self.report_times.append(time)

    # -- wells
    def well(self, record, name_item=0, create=False):
        name = record.text(name_item)
        if name is None:
            _fail(record.where, "missing_value", f"{record.keyword}: the well name is needed")
        if "*" in name or "?" in name:
            found = [w for w in self.wells if fnmatch.fnmatchcase(w, name)]
            if not found:
                _fail(record.where, "unknown_well", f"{record.keyword}: no well matches {name!r}", wells=list(self.wells))
            return [self.wells[w] for w in found]
        if name not in self.wells:
            _fail(record.where, "unknown_well", f"{record.keyword}: well {name!r} is not defined by WELSPECS",
                  wells=list(self.wells), did_you_mean=[w for w in self.wells if w.upper() == name.upper()])
        return [self.wells[name]]

    def do_welspecs(self, record):
        name = record.text(0)
        i, j = record.integer(2, "I"), record.integer(3, "J")
        if name in self.wells:
            _fail(record.where, "bad_value", f"WELSPECS: well {name!r} is defined twice")
        self.wells[name] = dict(name=name, i=i, j=j, order=len(self.wells), kind=None, blocks=[], radius=None, skin=None,
                                min_bhp=None, mode=None, values={}, status="OPEN", history=[], where=record.where)

    def do_compdat(self, record):
        nx, ny, nz = self.dims
        for well in self.well(record):
            i = record.integer(1, "I", well["i"]) if record.get(1) is not None else well["i"]
            j = record.integer(2, "J", well["j"]) if record.get(2) is not None else well["j"]
            k1 = record.integer(3, "K upper")
            k2 = record.integer(4, "K lower", k1) if record.get(4) is not None else k1
            if record.upper(5, "OPEN") not in ("OPEN", "AUTO"):
                self.warn(record.where, "shut_completion", f"COMPDAT: the shut completion of {well['name']} "
                          f"in layers {k1}-{k2} is left out")
                continue
            if record.get(7) is not None:
                self.warn(record.where, "connection_factor_ignored", f"COMPDAT: the connection factor of {well['name']} is "
                          "ignored; kusim computes the well index from the permeability, block size, radius and skin")
            direction = record.upper(12, "Z")
            if direction != "Z":
                _fail(record.where, "unsupported_feature", f"COMPDAT: well {well['name']} has direction {direction}: "
                      "only vertical completions (Z) are supported")
            radius = record.number(8, "diameter") / 2.0 if record.get(8) is not None else DEFAULT_WELL_RADIUS
            skin = record.number(10, "skin", 0.0) if record.get(10) is not None else 0.0
            if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k1 <= k2 <= nz):
                _fail(record.where, "bad_value", f"COMPDAT: well {well['name']} at ({i}, {j}, {k1}-{k2}) is outside the "
                      f"{nx} x {ny} x {nz} grid")
            for k in range(k1, k2 + 1):
                block = (k - 1) * nx * ny + (j - 1) * nx + i
                if block in well["blocks"]:
                    _fail(record.where, "bad_value", f"COMPDAT: well {well['name']} is completed twice in block ({i}, {j}, {k})")
                well["blocks"].append(block)
            if well["radius"] is None:
                well["radius"], well["skin"] = radius, skin
            elif abs(well["radius"] - radius) > 1e-12 or abs(well["skin"] - skin) > 1e-12:
                self.warn(record.where, "well_radius_differs", f"COMPDAT: well {well['name']} has completions with different "
                          "diameters or skins; kusim has one radius and one skin per well (the first one is used)")

    def _control(self, well, kind, where):
        if well["kind"] not in (None, kind):
            _fail(where, "bad_value", f"well {well['name']} is both a producer and an injector")
        well["kind"] = kind

    def _note_status(self, record, well, status):
        if status is not None and status not in ("OPEN", "SHUT", "STOP", "AUTO"):
            _fail(record.where, "bad_value", f"{record.keyword}: status must be OPEN, SHUT or STOP, got {status!r}")
        if status is not None:
            well["status"] = "SHUT" if status in ("SHUT", "STOP") else "OPEN"

    def do_wconprod(self, record):
        for well in self.well(record):
            self._control(well, "producer", record.where)
            self._note_status(record, well, record.upper(1, "OPEN"))
            mode = record.upper(2)
            known = {"ORAT": ("oil_rate", 3), "LRAT": ("liquid_rate", 6), "BHP": ("bhp", 8)}
            if mode not in known:
                _fail(record.where, "unsupported_value", f"WCONPROD: control mode {mode!r} of {well['name']} is not supported",
                      allowed=sorted(known), hint="kusim controls a producer by oil rate, liquid rate or bottom-hole pressure")
            for key, (m, item) in known.items():
                if record.get(item) is not None:
                    well["values"][m] = record.number(item, m)
            if record.get(4) is not None and record.number(4, "water rate") > 0:
                self.warn(record.where, "water_rate_limit_ignored", f"WCONPROD: the water rate limit of {well['name']} is ignored")
            if mode == "BHP" and "bhp" not in well["values"]:
                well["values"]["bhp"] = 14.7
            bhp_limit = record.get(8)
            if mode != "BHP" and bhp_limit is not None:
                if well["min_bhp"] is not None and abs(well["min_bhp"] - float(bhp_limit)) > 1e-9:
                    self.warn(record.where, "min_bhp_changes", f"WCONPROD: the bottom-hole pressure limit of {well['name']} "
                              "changes; kusim has one limit per well (the first one is used)")
                elif well["min_bhp"] is None:
                    well["min_bhp"] = float(bhp_limit)
            well["mode"] = known[mode][0]
            if well["mode"] not in well["values"]:
                _fail(record.where, "missing_value", f"WCONPROD: {well['name']} is controlled by {mode} but gives no value for it")
            self._emit(well)

    def do_wconhist(self, record):
        for well in self.well(record):
            self._control(well, "producer", record.where)
            self._note_status(record, well, record.upper(1, "OPEN"))
            mode = record.upper(2, "ORAT")
            oil, water = record.number(3, "oil rate", 0.0), record.number(4, "water rate", 0.0)
            if mode == "ORAT":
                well["mode"], well["values"]["oil_rate"] = "oil_rate", oil
            elif mode == "LRAT":
                well["mode"], well["values"]["liquid_rate"] = "liquid_rate", oil + water
            else:
                _fail(record.where, "unsupported_value", f"WCONHIST: control mode {mode!r} of {well['name']} is not supported",
                      allowed=["ORAT", "LRAT"], hint="the measured rates are imposed as oil rate or as liquid rate")
            self._emit(well)

    def do_wconinje(self, record):
        for well in self.well(record):
            self._control(well, "injector", record.where)
            fluid = record.upper(1)
            if fluid != "WATER":
                _fail(record.where, "unsupported_feature", f"WCONINJE: well {well['name']} injects {fluid}: only WATER is supported")
            self._note_status(record, well, record.upper(2, "OPEN"))
            mode = record.upper(3)
            if mode not in ("RATE", "BHP"):
                _fail(record.where, "unsupported_value", f"WCONINJE: control mode {mode!r} of {well['name']} is not supported",
                      allowed=["RATE", "BHP"], hint="kusim controls an injector by surface water rate or bottom-hole pressure")
            if record.get(4) is not None:
                well["values"]["water_rate"] = record.number(4, "rate")
            if record.get(6) is not None:
                well["values"]["bhp"] = record.number(6, "bhp")
            if mode == "RATE" and record.get(6) is not None:
                self.warn(record.where, "injector_bhp_limit_ignored", f"WCONINJE: the bottom-hole pressure limit of "
                          f"{well['name']} is ignored: kusim has no limit on an injector controlled by rate")
            well["mode"] = "water_rate" if mode == "RATE" else "bhp"
            if well["mode"] not in well["values"]:
                _fail(record.where, "missing_value", f"WCONINJE: {well['name']} is controlled by {mode} but gives no value for it")
            self._emit(well)

    def do_weltarg(self, record):
        for well in self.well(record):
            key = record.upper(1)
            value = record.number(2, "new value")
            names = ({"ORAT": "oil_rate", "LRAT": "liquid_rate", "BHP": "bhp"} if well["kind"] == "producer"
                     else {"RATE": "water_rate", "WRAT": "water_rate", "BHP": "bhp"})
            if well["kind"] is None or key not in names:
                _fail(record.where, "unsupported_value", f"WELTARG: {key!r} for well {well['name']} is not supported",
                      allowed=sorted(names), hint="WELTARG comes after the control keyword of the well")
            mode = names[key]
            well["values"][mode] = value
            if mode == well["mode"]:
                self._emit(well)
            elif mode == "bhp" and well["kind"] == "producer":
                self.warn(record.where, "min_bhp_changes", f"WELTARG: the new bottom-hole pressure limit of {well['name']} "
                          "is ignored (kusim has one limit per well)")

    def do_welopen(self, record):
        for well in self.well(record):
            status = record.upper(1, "OPEN")
            if any(record.get(i) is not None for i in (2, 3, 4, 5, 6)):
                _fail(record.where, "unsupported_feature", "WELOPEN: opening or shutting single connections is not supported",
                      hint="shut or open the whole well (leave items 3 to 7 defaulted)")
            self._note_status(record, well, status)
            if well["kind"] is not None:
                self._emit(well)

    def _emit(self, well):
        """Record the effective control of the well from now on."""
        if well["status"] == "SHUT":
            control = ("liquid_rate" if well["kind"] == "producer" else "water_rate", 0.0)
        else:
            control = (well["mode"], well["values"].get(well["mode"]))
        history = well["history"]
        if history and history[-1][0] == self.now:
            history.pop()
        if history and history[-1][1:] == control:
            return
        history.append((self.now, *control))


# ---------------------------------------------------------------------- building the model
def _per_axis(array, shape, axis, name, where):
    """The block sizes along one axis of a per-block array, which must not change in the other two directions."""
    view = array.reshape(shape)
    along = np.moveaxis(view, axis, 0).reshape(shape[axis], -1)
    if not np.allclose(along, along[:, :1], rtol=1e-9, atol=0):
        _fail(where, "unsupported_geometry", f"{name} changes in the other directions: kusim's grid has one block size per "
              f"{'column' if axis == 2 else 'row' if axis == 1 else 'layer'} ({'DX' if axis == 2 else 'DY' if axis == 1 else 'DZ'} "
              "may only depend on I, J or K respectively)")
    return along[:, 0].tolist()


def _need(reader, table, name, hint):
    if table not in reader.tables and table not in reader.single:
        _fail(str(reader.path.name), "missing_keyword", f"{name} is missing", hint=hint)


def _all_ones(reader, name):
    """ACTNUM, NTG and the region numbers must be 1 everywhere: kusim has no inactive blocks, net-to-gross or regions."""
    if name in reader.arrays:
        a = reader.arrays[name]
        if np.isnan(a).any() or not np.all(a == 1):
            _fail(reader.array_where.get(name, str(reader.path.name)), "unsupported_feature", f"{name} is not all 1: kusim has one region, no inactive "
                  "blocks and no net-to-gross", hint="fold net-to-gross into the porosity and permeability; remove inactive "
                  "blocks and regions")


def read_deck(path, dt=None, skip_unknown=False, pvt_pressures=None) -> DeckImport:
    """Read an ECLIPSE-style deck (see the module docstring). ``dt`` (days) must divide every report and event time;
    the default is their greatest common divisor. ``pvt_pressures`` (psi) is the pressure grid of the tables when the oil is
    given by ``PVCDO`` (default: 16 points between a quarter and twice the reference pressure)."""
    r = _Reader(path, dt, skip_unknown, pvt_pressures)
    r.read()
    where = r.path.name
    if r.units != "FIELD":
        _fail(where, "units_not_declared", "FIELD is not declared: kusim works in oilfield units (ECLIPSE assumes METRIC without "
              "a unit keyword)", hint="add FIELD to RUNSPEC (the deck must really be in field units)")
    if r.dims is None:
        _fail(where, "missing_keyword", "DIMENS is missing", hint="DIMENS gives the number of blocks in x, y and z")
    nx, ny, nz = r.dims
    n = nx * ny * nz
    shape = (nz, ny, nx)
    for name in ("ACTNUM", "NTG", *sorted(REGION_ARRAYS)):
        _all_ones(r, name)
    for name, array in r.arrays.items():
        if name in ("PORO", "PERMX", "PERMY", "PERMZ", "DX", "DY", "DZ", "TOPS") and np.isnan(array[:nx * ny if name == "TOPS" else n]).any():
            _fail(where, "bad_value", f"{name} is not defined in every block (an EQUALS box that does not cover the grid?)")

    # grid
    def sizes(axis_name, vector, axis, count):
        if vector in r.vectors:
            values = r.vectors[vector]
            if len(values) != count or any(not isinstance(v, float) for v in values):
                _fail(where, "bad_value", f"{vector} needs {count} numbers, has {len(values)}")
            return [float(v) for v in values]
        if axis_name not in r.arrays:
            _fail(where, "missing_keyword", f"{axis_name} (or {vector}) is missing", hint="block sizes in ft")
        return _per_axis(r.arrays[axis_name], shape, axis, axis_name, where)

    dx, dy, dz = sizes("DX", "DXV", 2, nx), sizes("DY", "DYV", 1, ny), sizes("DZ", "DZV", 0, nz)
    if "TOPS" not in r.arrays:
        _fail(where, "missing_keyword", "TOPS is missing", hint="TOPS gives the depth (ft) of the top of the first layer")
    top = r.arrays["TOPS"][:nx * ny]
    top_depth = float(top[0]) if np.allclose(top, top[0]) else top
    grid = Grid(dx, dy, dz, top_depth=top_depth)
    r.mapping.append(f"DIMENS/DX/DY/DZ/TOPS -> grid {nx} x {ny} x {nz}, top depth "
                     f"{top_depth if np.isscalar(top_depth) else 'varies by column'} ft")

    for name in ("PORO", "PERMX", "PERMY", "PERMZ"):
        if name not in r.arrays:
            _fail(where, "missing_keyword", f"{name} is missing",
                  hint="COPY PERMX PERMY / and COPY PERMX PERMZ / (with a MULTIPLY) are the usual way" if name != "PORO" else "")
    poro, kx, ky, kz = (r.arrays[k].copy() for k in ("PORO", "PERMX", "PERMY", "PERMZ"))
    r.mapping.append("PORO, PERMX, PERMY, PERMZ -> porosity, kx, ky, kz")

    # fluids
    for key, name, hint in (("PVTW", "PVTW", "water: reference pressure, Bw, compressibility, viscosity, viscosibility"),
                            ("ROCK", "ROCK", "reference pressure and rock compressibility"),
                            ("DENSITY", "DENSITY", "oil, water and gas density in lb/ft3")):
        _need(r, key, name, hint)
    if "PVDO" in r.tables:
        pvdo = r.tables["PVDO"]
        pressure, bo, mu_o = pvdo[:, 0], pvdo[:, 1], pvdo[:, 2]
        if "PVCDO" in r.single:
            _fail(where, "bad_value", "both PVDO and PVCDO are given")
    elif "PVCDO" in r.single:
        rec = r.single["PVCDO"]
        pref, bo_ref, co, mu_ref, cv = (rec.number(0, "p ref"), rec.number(1, "Bo"), rec.number(2, "compressibility"),
                                        rec.number(3, "viscosity"), rec.number(4, "viscosibility", 0.0) if rec.get(4) is not None else 0.0)
        pressure = np.asarray(pvt_pressures, float) if pvt_pressures is not None else np.linspace(0.25 * pref, 2.0 * pref, 16)
        x, y = co * (pressure - pref), (co - cv) * (pressure - pref)
        bo, mu_o = bo_ref / (1 + x + x * x / 2), mu_ref / (1 + y + y * y / 2)
        r.warn(rec.where, "pressure_grid_chosen", f"PVCDO is turned into a table on {len(pressure)} pressures between "
               f"{pressure[0]:g} and {pressure[-1]:g} psi (pvt_pressures / --pvt-pressures sets them)")
    else:
        _fail(where, "missing_keyword", "PVDO (or PVCDO) is missing", hint="oil Bo and viscosity against pressure")
    rec = r.single["PVTW"]
    pref, bw_ref, cw, mu_w_ref = rec.number(0, "p ref"), rec.number(1, "Bw"), rec.number(2, "compressibility"), rec.number(3, "viscosity")
    cv = rec.number(4, "viscosibility", 0.0) if rec.get(4) is not None else 0.0
    x, y = cw * (pressure - pref), (cw - cv) * (pressure - pref)
    bw, mu_w = bw_ref / (1 + x + x * x / 2), mu_w_ref / (1 + y + y * y / 2)
    r.mapping.append("PVDO/PVCDO, PVTW -> pvt table (Bo, Bw, oil and water viscosity against pressure)")
    rock = r.single["ROCK"]
    r.mapping.append(f"ROCK -> rock_compressibility {rock.number(1, 'compressibility'):g} 1/psi "
                     "(the reference pressure is not used: kusim keeps porosity constant)")
    dens = r.single["DENSITY"]
    gamma_o, gamma_w = dens.number(0, "oil density") / WATER_DENSITY, dens.number(1, "water density") / WATER_DENSITY
    r.mapping.append(f"DENSITY -> gamma_o {gamma_o:.4g}, gamma_w {gamma_w:.4g} (density / {WATER_DENSITY} lb/ft3)")

    if "SWOF" in r.tables:
        t = r.tables["SWOF"]
        sw, krw, kro, pcow = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    elif "SWFN" in r.tables and "SOF2" in r.tables:
        t, so = r.tables["SWFN"], r.tables["SOF2"]
        sw, krw, pcow = t[:, 0], t[:, 1], t[:, 2]
        kro = np.interp(1.0 - sw, so[:, 0], so[:, 1])
    else:
        _fail(where, "missing_keyword", "SWOF (or SWFN with SOF2) is missing", hint="relative permeability and capillary pressure")
    if np.any(np.diff(sw) <= 0):
        _fail(where, "bad_value", "the saturation table must increase")
    zero = np.nonzero(kro <= 0.0)[0]
    swc, sor = float(sw[0]), round(float(1.0 - (sw[zero[0]] if len(zero) else sw[-1])), 12)
    r.mapping.append(f"SWOF/SWFN+SOF2 -> rock_fluid table, swc {swc:g} (first saturation), sor {sor:g} "
                     "(where kro reaches 0)")

    # initial state
    if "EQUIL" in r.single:
        if "PRESSURE" in r.arrays or "SWAT" in r.arrays:
            _fail(where, "bad_value", "EQUIL and PRESSURE/SWAT are both given")
        e = r.single["EQUIL"]
        if e.number(3, "capillary pressure at the water-oil contact", 0.0) not in (0.0,) and e.get(3) is not None \
                and abs(e.number(3, "")) > 0:
            _fail(e.where, "unsupported_value", "EQUIL: a nonzero capillary pressure at the water-oil contact is not supported "
                  "(kusim takes zero there)", hint="set item 4 to 0")
        initial = Equilibrium(datum_pressure=e.number(1, "datum pressure"), woc_depth=e.number(2, "water-oil contact"),
                              datum_depth=e.number(0, "datum depth"))
        r.mapping.append("EQUIL -> initial equilibrium: datum depth, datum pressure and water-oil contact keep their values")
        if not np.any(pcow != 0.0):
            r.warn(e.where, "zero_capillary_pressure", "the capillary pressure table is all zero: the initial water saturation is "
                   "then a step at the water-oil contact (connate water above, the table's top saturation below)")
    elif "PRESSURE" in r.arrays and "SWAT" in r.arrays:
        p, s = r.arrays["PRESSURE"], r.arrays["SWAT"]
        if np.isnan(p).any() or np.isnan(s).any() or not (np.ptp(p) <= 1e-9 * max(abs(p[0]), 1) and np.ptp(s) <= 1e-12):
            _fail(where, "unsupported_feature", "PRESSURE and SWAT are not uniform: kusim starts from capillary-gravity "
                  "equilibrium (EQUIL) or from one pressure and one saturation", hint="use EQUIL")
        initial = UniformStart(pressure=float(p[0]), sw=float(s[0]))
        r.mapping.append("PRESSURE, SWAT -> uniform initial state")
    else:
        _fail(where, "missing_keyword", "EQUIL (or PRESSURE and SWAT) is missing", hint="the initial state")

    # wells and time
    if not r.report_times:
        _fail(where, "no_time_steps", "the deck has no TSTEP or DATES: the simulated time is not defined",
              hint="add TSTEP in SCHEDULE, for example TSTEP 60*10 /")
    t_end = r.report_times[-1]
    micro = lambda t: int(round(t * 1e6))
    times = sorted({*r.report_times, *(h[0] for w in r.wells.values() for h in w["history"] if 0 < h[0] < t_end)})
    if dt is None:
        g = 0
        for t in times:
            g = math.gcd(g, micro(t))
        dt = g / 1e6
    else:
        bad = [t for t in times if abs(t / dt - round(t / dt)) > 1e-6]
        if bad:
            _fail(where, "dt_not_commensurate", f"dt = {dt:g} days does not divide the times {[round(t, 6) for t in bad[:6]]} "
                  "(report and control-change times must be multiples of dt)", times=[round(t, 6) for t in bad[:20]],
                  hint="choose dt as a common divisor, such as " + f"{math.gcd(*[micro(t) for t in times]) / 1e6:g}")
    r.mapping.append(f"TSTEP/DATES -> t_end {t_end:g} days, dt {dt:g} days")

    producers, injectors, names, events = [], [], {}, []
    for well in sorted(r.wells.values(), key=lambda w: w["order"]):
        if well["kind"] is None or not well["blocks"]:
            why = "has no control keyword (WCONPROD, WCONHIST or WCONINJE)" if well["kind"] is None else "has no open completion"
            r.warn(well["where"], "well_not_used", f"well {well['name']} {why}: it is not in the model")
            continue
        history = well["history"]
        first = history[0]
        initial_control = first[1:] if first[0] == 0 else (("liquid_rate" if well["kind"] == "producer" else "water_rate"), 0.0)
        later = [h for h in history if 0 < h[0] < t_end]
        if first[0] > 0:
            r.warn(well["where"], "well_starts_late", f"well {well['name']} gets its first control at day {first[0]:g}: it is "
                   "shut (rate 0) until then")
        radius = well["radius"] if well["radius"] is not None else DEFAULT_WELL_RADIUS
        skin = well["skin"] or 0.0
        if well["kind"] == "producer":
            kusim_name = f"P{len(producers) + 1}"
            producers.append(Producer(well["blocks"], initial_control[0], initial_control[1], radius, skin, well["min_bhp"]))
        else:
            kusim_name = f"I{len(injectors) + 1}"
            injectors.append(Injector(well["blocks"], initial_control[0], initial_control[1], radius, skin))
        names[well["name"]] = kusim_name
        events += [ControlEvent(t, kusim_name, mode, target) for t, mode, target in later]
        dropped = [h for h in history if h[0] >= t_end]
        if dropped:
            r.warn(well["where"], "control_after_end", f"well {well['name']}: controls from day {dropped[0][0]:g} on come after "
                   f"the last time step (day {t_end:g}) and are left out")
    if not (producers or injectors):
        _fail(where, "missing_keyword", "the deck has no wells with controls", hint="WELSPECS, COMPDAT and WCONPROD/WCONINJE")
    r.mapping.append("WELSPECS/COMPDAT/WCON*/WELTARG/WELOPEN -> wells " + ", ".join(f"{d} = {k}" for d, k in names.items())
                     + f" and {len(events)} schedule events")

    try:
        model = ReservoirModel(
            grid=grid, kx=kx, ky=ky, kz=kz, porosity=poro, rock_compressibility=rock.number(1, "compressibility"),
            pvt=PVTTable(pressure, bo, bw, mu_o, mu_w), rock_fluid=RockFluidTable(sw, kro, krw, pcow),
            gamma_o=gamma_o, gamma_w=gamma_w, swc=swc, sor=sor, initial=initial, dt=float(dt), t_end=float(t_end),
            producers=producers, injectors=injectors,
            provenance=[f"imported from {r.path.name}; wells " + ", ".join(f"{d}={k}" for d, k in names.items())])
    except ValueError as error:
        _fail(where, "invalid_model", f"the deck gives a model that kusim rejects: {error}")
    return DeckImport(model, Schedule(events), names, r.warnings, r.ignored, r.mapping)
