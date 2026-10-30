# Reading an ECLIPSE-style deck

```
kusim import-deck CASE.DATA -o model.json --schedule-out schedule.json --wells-out wells.json
```

writes a kusim JSON model (and the well-control changes after day 0 as a schedule) from a deck. From Python:
`kusim.read_deck("CASE.DATA")` returns the `model`, the `schedule`, the `wells` map and the `warnings`, `ignored` and
`mapping` lists that the command prints. A working example is in `examples/eclipse/` (`waterflood.DATA` and an `INCLUDE`d
`waterflood_props.INC`).

**The reader is strict.** kusim is an oil-water simulator on a Cartesian grid in oilfield units. A keyword or value that kusim cannot
honour stops the import with a located error (`"path": "CASE.DATA:57"`, a `code`, a `hint`) because skipping it would change the
answer without telling you. `--skip-unknown` turns those errors into warnings (the data of the skipped keyword is skipped up to
the next keyword the reader knows; use it only when you are sure the keyword does not matter). Keywords that have no effect on
the model (`TITLE`, `WELLDIMS`, `RPTSCHED`, the whole `SUMMARY` section, ...) are accepted and listed under `ignored`.
The answer's `mapping` says where each piece of the deck went.

## What is read

| Section | Keywords | Becomes |
|---|---|---|
| RUNSPEC | `DIMENS`, `FIELD` (required), `OIL`, `WATER`, `START` | grid size, the units check, the date of day 0 (for `DATES`) |
| GRID | `DX` `DY` `DZ` (one value per block) or `DXV` `DYV` `DZV` (one per column, row, layer), `TOPS` | `grid` (sizes may only depend on I, J and K respectively; a flat top or one value per column) |
| | `PORO`, `PERMX`, `PERMY`, `PERMZ` | `porosity`, `kx`, `ky`, `kz` |
| | `EQUALS`, `MULTIPLY`, `ADD`, `COPY` (with or without a box `I1 I2 J1 J2 K1 K2`) | edit those arrays |
| | `NTG`, `ACTNUM`, `SATNUM`, `PVTNUM`, `EQLNUM`, `FIPNUM` | accepted when they are 1 everywhere |
| PROPS | `PVDO` (pressure, Bo, viscosity) or `PVCDO` | oil `bo`, `mu_o` against pressure |
| | `PVTW` | `bw`, `mu_w` on the same pressures (ECLIPSE's compressibility formulas) |
| | `ROCK` | `rock_compressibility` (the reference pressure is not used: porosity is constant in kusim) |
| | `DENSITY` | `gamma_o`, `gamma_w` = density / 62.4 lb/ft³ |
| | `SWOF` or `SWFN` + `SOF2` | `rock_fluid` (Sw, krw, kro, Pcow); `swc` is the first saturation, `sor` is 1 minus the saturation where kro reaches 0 |
| SOLUTION | `EQUIL` | `initial`: equilibrium, with **`datum_depth`, `datum_pressure` and `woc_depth` taken as they are** (no change of reference) |
| | `PRESSURE` and `SWAT` | a uniform start, when the arrays are uniform |
| SCHEDULE | `WELSPECS`, `COMPDAT` | the wells, their blocks, radius (diameter / 2, default 0.5 ft) and skin |
| | `WCONPROD` (`ORAT`, `LRAT`, `BHP`), `WCONINJE` (`WATER`, `RATE` or `BHP`), `WCONHIST` (`ORAT`, `LRAT`) | the control of the well; later ones are schedule events |
| | `WELTARG`, `WELOPEN` | a new target of the active mode, a well shut (rate 0) or opened |
| | `TSTEP`, `DATES` | `t_end` and `dt`; the time of the control changes |
| anywhere | `INCLUDE 'file' /` | read as part of the deck (relative to the including file) |

## How it maps

- **Wells get kusim's names.** kusim names wells `P1, P2, ...` (producers) and `I1, ...` (injectors) in the order of `WELSPECS`.
  `wells` (and `--wells-out`) gives the deck names: `{"PROD-A": "P1"}`. `import-history` reads that file, so measured data
  carry over.
- **Depth and pressure.** The depth frame of the deck is kept: `TOPS` is `grid.top_depth`, and `EQUIL`'s datum depth is
  `initial.datum_depth` with its pressure. The model is then the same one the deck describes, in absolute depths.
- **Time.** `dt` is the greatest common divisor of the `TSTEP`/`DATES` intervals unless `--dt` is given (it must divide every
  report time and every time a control changes; a refusal names the times that do not fit). kusim takes fixed steps, ECLIPSE
  chooses its own: the import runs the first steps (`checks` in the answer, as `validate` does) and warns when `dt` is too large for
  the explicit saturation update, with a suggestion. Controls before the first `TSTEP` are the model's controls, later ones
  are events. A well whose first control comes later is shut (rate 0) until then.
- **A producer's `min_bhp`** is the bottom-hole pressure limit of `WCONPROD` (item 9) when the well is controlled by a rate.
- **`PVCDO`** is turned into a table on 16 pressures between a quarter and twice the reference pressure
  (`--pvt-pressures 1000,2000,...` sets them).

## What is refused (with a hint)

Gas, dissolved gas and live oil (`GAS`, `DISGAS`, `PVTO`, `PVTG`, ...), corner-point geometry (`COORD`, `ZCORN`), `METRIC` or a
missing `FIELD`, more than one region (several `SATNUM` values, a second `PVTW` record), a nonzero capillary pressure at the water-oil
contact in `EQUIL`, non-uniform `PRESSURE`/`SWAT`, `NTG` or `ACTNUM` not equal to 1, multipliers (`MULTX`, `MULTPV`), end-point scaling
(`SWL`, `SOWCR`, ...), aquifers, group controls, `BOX`/`ENDBOX`, a water-rate producer, a well control mode other than the ones above,
horizontal completions, and any keyword the reader does not know. The first error stops the import, so fix and run again.

## Differences from ECLIPSE to keep in mind

- The numerical scheme is different (explicit saturations, fixed steps, upstream weighting by potential): results will agree in
  trend, not to the digit (see `eclipse_comparison/`).
- The PVT interpolation differs (kusim interpolates 1/B and the viscosity, ECLIPSE 1/B and 1/(B·mu)); with a fine table the
  difference vanishes.
- Where ECLIPSE would use `COMPDAT`'s connection factor, kusim computes the Peaceman well index from the permeability, block size,
  radius and skin; a given connection factor is ignored with a warning.
- ECLIPSE defaults to METRIC when no unit keyword is given; kusim insists on `FIELD`.
