# Plotting from a description: `kusim tables`, `kusim plot-template`, `kusim plot`

A figure is described in a JSON file and drawn by `kusim plot`. The description is **data, not code**: nothing in it is executed.
It has three parts:

```json
{"sources": {"run": "run/results.json", "model": "model.json"},
 "tables":  {"avg": {"from": "run.snapshots", "where": [...], "group_by": [...], "aggregate": {...}, "pivot": {...}}},
 "figure":  {"data": [{"type": "heatmap", "z": {"$ref": "avg.z"}, ...}], "layout": {...}}}
```

1. **sources** give names to kusim files (a model, a run's `results.json` or `summary.json`, observations, the table of
   `kusim misfit --out`). File names are relative to the working folder; `..`, drive letters and absolute names are refused.
2. **tables** are queries over the tables that those files offer (below). They average, filter, join and pivot, so the figure
   never contains numbers typed by hand.
3. **figure** is a [Plotly figure](https://plotly.com/python/reference/) (`data` and `layout`) in which a value may be
   `{"$ref": "table.column"}`.

```
kusim plot-template                                  # the worked examples (what they draw, which files they need)
kusim plot-template --name average_map -o spec.json  # write one, change it, then:
kusim plot spec.json -o fig.png                      # also .svg .pdf .jpg .webp .html .json  (--width --height --scale)
kusim tables run/results.json                        # which tables and columns a file offers, with units
```

`kusim plot` prints one JSON object: `written`, the size, a summary of every table (`n_rows`, `columns`, the first rows, or the
`x`/`y`/`z` ranges of a pivoted one), the number and types of traces, and `warnings`. **Read the warnings**: a trace whose table
has no rows draws nothing (usually a `where` that matches no row). Errors say where (`path`, for example `figure.data.1.y` or
`tables.avg.where.0.column`), the `code`, what is available and `did_you_mean`.

## Tables of each kind of file (`kusim tables FILE`)

| File | Tables |
|---|---|
| model | `cells` (block, i, j, k, x, y, depth, dx, dy, dz, volume, pore_volume, kx, ky, kz, porosity), `wells` (well, kind, mode, target, target_unit, i, j, k_top, k_bottom, x, y, ...), `completions` (one row per perforated block) |
| `results.json` | `series` (time, vector, quantity, well, value: `WWCT:P1`, `WBHP:I1`, `FOPT`, ...), `wells_history` (one row per well and step), `perforations`, `snapshots` (time, block, i, j, k, x, y, depth, pressure, sw, so) |
| `summary.json` | `summary_wells` (one row per well: mode, bhp, rates, cumulative volumes) |
| observations | `observations` (time, vector, quantity, well, value, sigma, weight) |
| misfit table | `residuals` (time, vector, quantity, well, observed, simulated, residual, normalised, weight) |

x, y, depth are in ft (cell centres), pressure in psi, rates in STB/day, time in days. `snapshots` has one row per block **and saved
time**: select times in `where` (`{"column": "time", "op": "==", "value": "last"}`; a number, `"first"` and `"last"` work, and only those
snapshots are built). Run with `--fields last` (the default) for one snapshot or `--fields all` for every saved time; a map at a time
that was not stored is an empty table (the warning says so).

## Queries

A query is `{"from": ...}` plus any of these steps, applied **in this order**: `join`, `where`, `group_by` with `aggregate`, `sort_by`,
`select`, `limit`, `pivot`. A query may start from a source table (`"run.series"`) or from a table defined before it. A table name may not
be the name of a source or contain a dot.

| Step | Form |
|---|---|
| `where` | list of `{"column", "op", "value"}`; `op` is `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not_in`; a value may be `"first"` or `"last"` (smallest or greatest in that column) |
| `join` | `{"table": "model.cells", "on": "block" or [columns], "how": "inner"/"left", "suffix": "_right"}`; columns with the same name in both tables get the suffix on the right one |
| `group_by` + `aggregate` | `{"sw": {"fn": "mean", "column": "sw", "weight": "pore_volume"}}`; `fn` is `mean`, `sum`, `min`, `max`, `count`, `first`, `last`, `std`, `median`, `rms`; `weight` only with `mean` (pore-volume weighted average) |
| `sort_by`, `select`, `limit` | a column or list (`"-rms"` sorts down), the columns to keep, the number of rows |
| `pivot` | `{"rows": "y", "columns": "x", "values": "sw"}` turns the table into a grid, last step. Every (row, column) pair must occur once: group first, or filter to one layer and one time |

There is no free arithmetic. Residuals and normalised residuals come from the misfit table; averages, sums, RMS and counts from `aggregate`.

## Refs and one trace per group

`{"$ref": "avg.z"}` is the column `z` of table `avg`; for a pivoted table the parts are `.x`, `.y` and `.z` (a heatmap's `x`, `y`, `z`).
`{"$ref": "run.series.value"}` reads a source table directly. A trace with `"$each": {"table": "wc", "by": "well"}` is repeated for every
value of that column. Inside it, **every table that has the column** is cut to that value, so observed and simulated data can be drawn
well by well, and `"$group"` stands for the value (also inside a text: `"name": "observed $group"`).

## What a figure may not contain

The figure is only drawn, and kusim refuses what would reach outside it: trace types other than the usual ones (`scatter`, `scattergl`,
`bar`, `histogram`, `histogram2d`, `heatmap`, `contour`, `box`, `violin`, `scatter3d`, `mesh3d`, `surface`, `isosurface`, `volume`, `pie`,
`waterfall`, `funnel`, `image`: the map types fetch tiles), `layout.geo`, `layout.map`, `layout.mapbox`, any text with a link
(`://`, `javascript:`), and figures with more than 3 million numbers. Plotly validates every property: an unknown one is an error that names it.

## Worked examples

`kusim plot-template` lists them; they are also in `src/kusim/plot_examples/` and every one is run by `tests/test_plotting.py`.

| Name | Draws |
|---|---|
| `average_map` | water saturation averaged over the layers (pore-volume weighted) at the last time, wells labelled |
| `layer_map` | an input property (kx) of one layer, wells labelled |
| `water_cut_by_well` | simulated water cut against time, one line per producer |
| `observed_vs_simulated` | observed points with error bars over simulated lines, well by well |
| `vertical_profile` | average water saturation of every layer against depth |
| `mismatch_map` | RMS of the normalised residuals of every well, as symbols on the map |
| `mismatch_bars` | the same per well and quantity as bars |

Other things follow the same pattern: an average over time (`group_by` `x`, `y` over several snapshots), pressure instead of saturation,
a cross-section (`where` on `j`, then `x` and `depth`), cumulative production of two runs (two sources, two `series` tables), a histogram of the
residuals (`type: histogram` on the column).

## Limits

- Not yet: 3-D views and animations over time (the viewers in `kusim.viz` do that from Python), arithmetic on columns.
- Images (`.png`, `.svg`, ...) are drawn by kaleido, which needs a Chrome or Chromium browser (`pip install kusim[plot]`). Each call
  starts it: about 5 s, and the first one after installation can take much longer. `.html` (an interactive page that works offline,
  about 5 MB) and `.json` (the finished figure) need no browser.
- The agent of the tutorial cannot see the picture. It reports where the file is; look at it yourself, and compare the numbers in the
  answer (`tables`) with what you expect.
