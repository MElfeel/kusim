"""A plot description (JSON) -> a checked Plotly figure -> a PNG, SVG, PDF, HTML or JSON file. No code from the description is run.

The description has three parts (``docs/plotting.md`` has worked examples)::

    {"sources": {"run": "runs/a/results.json", "obs": "observations.json"},     # names for files (relative names only)
     "tables":  {"wc": {"from": "run.series", "where": [...], ...}},             # queries (kusim.queries)
     "figure":  {"data": [...], "layout": {...}}}                                # a Plotly figure

Inside ``figure`` a value may be ``{"$ref": "wc.value"}`` (the column ``value`` of the table ``wc``; for a pivoted table ``.x``, ``.y`` or
``.z``; ``source.table.column`` reads a source table directly). A trace may have ``"$each": {"table": "wc", "by": "well"}``: it is then
repeated for every value of that column; inside it, every table that has the column ``by`` only shows the rows of that value (so two tables, observed
and simulated, can be drawn well by well), and ``"$group"`` stands for the value (also inside a text: ``"observed $group"``). The trace is named after
the value unless it has a name.

What is refused, because the figure is only ever drawn: trace types that fetch map tiles or geographic data (only the types in ``TRACE_TYPES`` are
accepted), ``layout.geo``, ``layout.map`` and ``layout.mapbox``, any text that contains a URL, and figures with too many numbers.
"""

from __future__ import annotations

from pathlib import Path

from .queries import Matrix, jsonable, load_source, need_columns, run_tables, safe_relative, table_summary, _suggest
from .spec import SpecError, _check_keys

SPEC_KEYS = {"sources", "tables", "figure"}
TRACE_TYPES = ("scatter", "scattergl", "bar", "histogram", "histogram2d", "heatmap", "contour", "box", "violin", "scatter3d", "mesh3d", "surface",
               "isosurface", "volume", "pie", "waterfall", "funnel", "image")
FORBIDDEN_LAYOUT = ("geo", "map", "mapbox")
OUTPUT_FORMATS = (".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp", ".html", ".json")
MAX_NUMBERS = 3_000_000
MAX_PIXELS = 64_000_000
URL_MARKERS = ("://", "javascript:", "data:text/html")


# ---------------------------------------------------------------------- worked examples
EXAMPLE_DIR = Path(__file__).resolve().parent / "plot_examples"
EXAMPLE_INFO = {
    "average_map": ("Map of the water saturation averaged over the layers (pore-volume weighted) at the last saved time, wells labelled. "
                    "Change the column (sw, pressure), the time, or add a where on k to average some layers only.",
                    ["model.json", "run/results.json (kusim run --out run --fields last or all)"]),
    "layer_map": ("Map of one layer of an input property (permeability of layer 2), wells labelled. Change k and the column "
                  "(kx, ky, kz, porosity).", ["model.json"]),
    "water_cut_by_well": ("Simulated water cut against time, one line per producer. Change the quantity (WWCT, WOPR, WBHP, ...) in the where.",
                          ["run/results.json"]),
    "observed_vs_simulated": ("Observed points (with error bars of sigma) over simulated lines, well by well. Change the quantity in both wheres.",
                              ["run/results.json", "observations.json (kusim import-history)"]),
    "vertical_profile": ("Average water saturation of every layer against depth at the last time (a profile through the model).",
                         ["model.json", "run/results.json (--fields last or all)"]),
    "mismatch_map": ("Map of the mismatch of every well (RMS of residual/sigma of one quantity); symbol size and colour show the RMS.",
                     ["model.json", "misfit.json (kusim misfit ... --out misfit.json)"]),
    "mismatch_bars": ("Bar chart of the mismatch of every well and quantity (RMS of residual/sigma).", ["misfit.json (kusim misfit ... --out misfit.json)"]),
}


def example_spec(name):
    """The text of a worked example (raises ``SpecError`` ``unknown_example`` with the names)."""
    if name not in EXAMPLE_INFO:
        raise SpecError(f"there is no example {name!r}; the examples are {sorted(EXAMPLE_INFO)}", "name", "unknown_example",
                        available=sorted(EXAMPLE_INFO), did_you_mean=_suggest(name, EXAMPLE_INFO))
    return (EXAMPLE_DIR / f"{name}.json").read_text(encoding="utf8")


# ---------------------------------------------------------------------- binding
class _Binder:
    def __init__(self, tables):
        self.tables = tables
        self.numbers = 0
        self.empty = []

    def resolve(self, ref, path, group):
        if not isinstance(ref, str) or not ref:
            raise SpecError("$ref must be a text like 'table.column'", path, "bad_ref")
        parts = ref.split(".")
        if parts[0] in self.tables.derived:
            name, rest = parts[0], parts[1:]
            table = self.tables.derived[name]
        elif len(parts) >= 3 and parts[0] in self.tables.sources:
            name, rest = f"{parts[0]}.{parts[1]}", parts[2:]
            table = self.tables.get(name, path)
        else:
            raise SpecError(f"$ref {ref!r}: there is no table {parts[0]!r}; tables are {sorted(self.tables.derived)} (a source table is written "
                            "source.table.column)", path, "unknown_ref", tables=sorted(self.tables.derived),
                            did_you_mean=_suggest(parts[0], list(self.tables.derived)))
        if len(rest) != 1:
            raise SpecError(f"$ref {ref!r} must be table.column (one column or, for a pivoted table, x, y or z)", path, "bad_ref")
        part = rest[0]
        if isinstance(table, Matrix):
            if part not in ("x", "y", "z"):
                raise SpecError(f"$ref {ref!r}: table {name!r} was pivoted into a grid; use {name}.x, {name}.y or {name}.z", path, "unknown_ref")
            values = table.part(part)
            self.numbers += len(values) * (len(table.x) if part == "z" else 1)
            return values
        need_columns(table, [part], path, name)
        if group is not None and group[0] in table.columns:             # one trace per group: every table that has the column is cut to it
            table = table[table[group[0]] == group[1]]
        values = jsonable(table[part])
        if not values:
            self.empty.append({"path": path, "ref": ref})
        self.numbers += len(values)
        return values

    def bind(self, node, path, group=None):
        if isinstance(node, dict):
            if "$ref" in node:
                if len(node) != 1:
                    raise SpecError("a {\"$ref\": ...} object may not have other keys", path, "bad_ref")
                return self.resolve(node["$ref"], path, group)
            return {k: self.bind(v, f"{path}.{k}", group) for k, v in node.items()}
        if isinstance(node, list):
            return [self.bind(v, f"{path}.{n}", group) for n, v in enumerate(node)]
        if isinstance(node, str) and "$group" in node:
            if group is None:
                raise SpecError("'$group' can only be used inside a trace that has '$each'", path, "bad_ref")
            return group[1] if node == "$group" else node.replace("$group", str(group[1]))
        return node

    def traces(self, data, path):
        if not isinstance(data, list):
            raise SpecError("figure.data must be a list of traces", path, "bad_value")
        out = []
        for n, trace in enumerate(data):
            here = f"{path}.{n}"
            if not isinstance(trace, dict):
                raise SpecError("a trace must be an object", here, "bad_value")
            if "$each" not in trace:
                out.append(self.bind(trace, here))
                continue
            each = trace["$each"]
            _check_keys("$each", each, {"table", "by"}, {"table", "by"}, path=f"{here}.$each")
            table = self.tables.derived.get(each["table"])
            if table is None or isinstance(table, Matrix):
                raise SpecError(f"$each needs a table of the 'tables' part, got {each['table']!r}; tables are {sorted(self.tables.derived)}",
                                f"{here}.$each.table", "unknown_ref", tables=sorted(self.tables.derived))
            need_columns(table, [each["by"]], f"{here}.$each.by", each["table"])
            body = {k: v for k, v in trace.items() if k != "$each"}
            for value in jsonable(table[each["by"]].drop_duplicates()):
                bound = self.bind(body, here, (each["by"], value))
                bound.setdefault("name", str(value))
                out.append(bound)
        return out


# ---------------------------------------------------------------------- checks
def _strings(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        if node and not isinstance(node[0], (str, list, dict)):
            return                                                  # a long list of numbers: nothing to scan
        for n, v in enumerate(node):
            yield from _strings(v, f"{path}.{n}")
    elif isinstance(node, str):
        yield path, node


def check_figure(figure, path="figure"):
    """Raise ``SpecError`` for anything the figure may not contain (see the module docstring)."""
    _check_keys("the figure", figure, {"data", "layout", "frames", "config"}, {"data"}, path=path)
    for n, trace in enumerate(figure["data"]):
        kind = trace.get("type", "scatter")
        if kind not in TRACE_TYPES:
            raise SpecError(f"trace type {kind!r} is not accepted (it may fetch map tiles or other files); accepted types: {list(TRACE_TYPES)}",
                            f"{path}.data.{n}.type", "forbidden_content", allowed=list(TRACE_TYPES), did_you_mean=_suggest(kind, TRACE_TYPES))
    layout = figure.get("layout", {})
    if not isinstance(layout, dict):
        raise SpecError("figure.layout must be an object", f"{path}.layout", "bad_value")
    for key in FORBIDDEN_LAYOUT:
        if key in layout:
            raise SpecError(f"layout.{key} is not accepted: it fetches map tiles or geographic data", f"{path}.layout.{key}", "forbidden_content")
    for where, text in _strings(figure, path):
        low = text.lower()
        if any(m in low for m in URL_MARKERS):
            raise SpecError(f"the text at {where} contains a link ({text[:60]!r}): a figure may not refer to anything outside it",
                            where, "forbidden_content")


def build_figure(figure):
    """A validated ``plotly.graph_objects.Figure`` (Plotly's own checks: an unknown property is an error that names it)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SpecError("plotting needs the plotly package (and kaleido for images): pip install kusim[plot]", "figure", "missing_dependency") from None
    try:
        return go.Figure(figure)
    except (ValueError, TypeError) as error:
        raise SpecError(f"the figure is not valid: {error}", "figure", "invalid_figure") from None


# ---------------------------------------------------------------------- the whole description
def prepare(spec, spec_path="spec"):
    """``(figure_dict, tables, info)`` from a plot description: sources read, queries run, refs bound, figure checked."""
    _check_keys("the plot description", spec, SPEC_KEYS, {"figure"})
    sources = {}
    declared = spec.get("sources", {})
    if not isinstance(declared, dict):
        raise SpecError("sources must be an object: name -> file name", "sources", "not_an_object")
    for name, file in declared.items():
        if not name or "." in name:
            raise SpecError(f"a source name may not be empty or contain '.', got {name!r}", f"sources.{name}", "bad_value")
        sources[name] = load_source(name, file, f"sources.{name}")
    tables = run_tables(sources, spec["tables"]) if "tables" in spec else run_tables(sources, {})
    binder = _Binder(tables)
    figure = spec["figure"]
    if not isinstance(figure, dict) or "data" not in figure:
        raise SpecError("figure must be a Plotly figure object with a 'data' list", "figure", "missing_key")
    bound = {k: (binder.traces(v, "figure.data") if k == "data" else binder.bind(v, f"figure.{k}")) for k, v in figure.items()}
    if binder.numbers > MAX_NUMBERS:
        raise SpecError(f"the figure holds about {binder.numbers} numbers (over {MAX_NUMBERS}): aggregate or filter the tables", "figure", "too_large")
    check_figure(bound)
    info = {"tables": {name: table_summary(t) for name, t in tables.derived.items()},
            "figure": {"n_traces": len(bound["data"]), "trace_types": [t.get("type", "scatter") for t in bound["data"]]}}
    warnings = [f"{e['path']}: {e['ref']} has no rows, so this trace draws nothing (check the where conditions and the names)" for e in binder.empty]
    if warnings:
        info["warnings"] = warnings
    return bound, tables, info


def render(bound, output, width=900, height=600, scale=2.0):
    """Write the figure; the extension of ``output`` chooses the format. Returns what was written."""
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix not in OUTPUT_FORMATS:
        raise SpecError(f"the output file must end in one of {list(OUTPUT_FORMATS)}, got {output.name!r}", "output", "bad_value", allowed=list(OUTPUT_FORMATS))
    if not (1 <= width <= 4000 and 1 <= height <= 4000 and 0 < scale <= 4 and width * height * scale * scale <= MAX_PIXELS):
        raise SpecError("width and height must be between 1 and 4000 pixels, scale at most 4, and the image at most 64 million pixels",
                        "output", "too_large")
    fig = build_figure(bound)
    if "width" not in bound.get("layout", {}) and "height" not in bound.get("layout", {}):
        fig.update_layout(width=width, height=height)
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        fig.write_json(str(output))
    elif suffix == ".html":
        fig.write_html(str(output), include_plotlyjs=True)          # the plotly library is embedded: the file works offline
    else:
        try:
            fig.write_image(str(output), width=width, height=height, scale=scale)
        except Exception as error:                                   # kaleido or its browser is missing or failed
            raise SpecError(f"the image could not be rendered ({type(error).__name__}: {str(error)[:300]}). kaleido needs a Chrome or Chromium "
                            "browser; write a .html or .json file instead, or ask for help", "output", "render_failed") from None
    return {"written": str(output), "format": suffix[1:], "width": width, "height": height, "bytes": output.stat().st_size}
