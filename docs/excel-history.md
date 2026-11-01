# Historical data from Excel

```
kusim history-template -o history.xlsx            # an example workbook to see the layout
kusim import-history history.xlsx -o observations.json --model model.json --sigma WBHP=25 --sigma WWCT=0.02 \
      [--controls-out schedule.json] [--wells-file wells.json | --well-map PROD-A=P1] [--start-date 2020-01-01]
```

reads measured data from a workbook (`.xlsx`, `.xlsm`) or a CSV file and writes kusim **observations**
(`time`, `vector`, `value`, `sigma`, `weight`: [json-format.md](json-format.md#observations)), and, if asked, the measured rates as
well **controls**. From Python: `kusim.read_history("history.xlsx", model=model, ...)` returns `observations`, `schedule` and the
`report`. Excel support needs `pip install kusim[excel]` (openpyxl).

## What the workbook looks like

One header row (`--header-row N` if it is lower) and one sheet (`--sheet`, default the first). Three layouts, found from the
headers (`--layout` forces one):

**`by_well`**, the usual database export: one row per date and well, one column per quantity.

| Date | Well | Oil rate (STB/d) | Water rate (STB/d) | BHP (psi) | Water injection (STB/d) |
|---|---|---|---|---|---|
| 2020-02-01 | P1 | 2950 | 35 | 1620 | |
| 2020-02-01 | I1 | | | 3010 | 6000 |

**`wide`**: one row per date, one column per well and quantity: `WBHP:P1`, `P1 BHP`, `BHP P1`, `oil rate P1`, `WWIR_I1`.

**`long`**: one row per measurement: `time` (or `date`), `vector` (`WBHP:P1`, or a quantity with a `well` column), `value`, and
optionally `sigma` and `weight`.

- **Time** is a `Time`/`Days` column (days since the start) or a `Date` column; days are counted from `--start-date` (default: the
  first date, which is then day 0 and dropped). Dates like `03/04/2020` are read month first unless `--dayfirst`.
- **Quantities** are recognised by name (case, spaces and underscores do not matter): oil rate (`WOPR`, `Qo`), water rate (`WWPR`,
  `Qw`), liquid rate, BHP (`Pwf`, `pressure`), water cut (`WWCT`, `WC`), water injection (`WWIR`, `injection rate`). A column that
  matches nothing is listed under `columns_ignored`. For an injector, `water rate` means `WWIR`.
- **Units** come from the header (`BHP (bar)`, `Oil rate (m3/d)`, `Water cut (%)`) or from `--pressure-unit`, `--rate-unit`,
  `--water-cut-unit` (psi, bar, kPa, MPa, atm, kgf/cm2; stb/d, bbl/d, m3/d; fraction, percent). Everything is converted to oilfield units.
  A water cut above 1 is refused ("is it in percent?").
- A **negative water cut** (noise around zero) is set to 0 and reported under `trimmed`; other negative values are dropped and counted.
- **Blank cells** (and `n/a`, `NA`, which Excel/pandas read as blank) are skipped; text or other negative numbers are dropped and counted.

## Wells

Names in the sheet must be kusim's (`P1`, `I1`) or mapped: `--well-map PROD-A=P1` (repeatable) or `--wells-file wells.json`, the
`--wells-out` of `import-deck`. With `--model`, the wells and their kind are checked (an oil rate for an injector is refused, a well the
model does not have is refused with the list it has).

## Sigma: the measurement error

`sigma` is the standard deviation of the measurement error in the unit of the vector (psi, a fraction, STB/d). It decides how
much a data point counts: the objective is the sum of `((simulated - observed) / sigma)²`. Give it where you know it:

- a `sigma` column in a `long` sheet (per point), or
- `--sigma WBHP=25 --sigma WWCT=0.02 --sigma WOPR=5%` (absolute, or a percentage of the value with a floor of 1 % of the largest value),
- otherwise defaults (WBHP 25 psi, WWCT 0.02, rates 5 %) are used, and **the answer says so** under `defaults_used` and `warnings`.

Choose sigma to include the model error, not only the gauge accuracy; a too small sigma makes an ensemble collapse.

## Matching the time step

kusim runs fixed steps and compares at times that are multiples of `dt`. With `--model` (or `--dt`) the data of each interval of
`--every` days (default `dt`) become **one observation at the end of the interval**: the mean (`--align last`: the last value).
Data before day 0 or after the model's end time are dropped. Without `dt` the times are kept as they are, with a warning.

## Measured rates as controls

`--controls-out schedule.json` also writes the measured rates as a schedule: producers by liquid rate (oil + water, or a liquid column)
or `--producer-control oil_rate`, injectors by water rate. Each interval's rate acts **from the start of the interval** in which it was
measured; a rate that does not change makes no event; where rates are missing, the last one stays in force (a warning says so). Do not
also match the rates you impose: match the pressures and the water cuts (`--vectors WBHP,WWCT`). The first interval's control is an
event at day 0; give `import-deck`'s model the same initial control, or let the event take over.

## The report

The answer lists `layout`, `time`, `wells`, `units`, `observations` (count per vector and time range), `dropped` (a count and the
first Excel rows for each reason), `defaults_used`, `warnings`, `columns_ignored` and what was written. Check `dropped` and
`defaults_used` before you trust a calibration made on the data.

Errors are located and carry hints, like everywhere (`unmapped_well` lists the sheet's names; `unknown_sheet` the sheets;
`unknown_unit`, `missing_column`, `unknown_vector`, `bad_vector_for_well`, `no_data`, `unreadable_file`).
