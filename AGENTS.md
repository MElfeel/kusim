# Using kusim from an agent or a script

kusim is a 3-D, two-phase (oil/water) reservoir simulator. This page is for a program or an LLM agent that **drives it
through files and the command line**. Developers of the package: see `CLAUDE.md`. The file formats are in
`docs/json-format.md`; this page says what to run, what to check and what not to believe.

## Ground rules

1. **Talk to kusim with JSON files and `kusim ...` commands.** Every command prints exactly **one JSON object** on
   standard output. Look at `"ok"` first (the exceptions are `kusim schema NAME` and `kusim template` without `-o`, which
   print the document itself; their errors still have `"ok": false`). Exit code: `0` success, `2` bad input (the object has `error`, usually also
   `code`, `path` and hints), `1` only with `--strict` when the run finished but is not healthy.
2. **Units are oilfield units**: ft, md, psi, cp, STB/day, days, 1/psi. Rates are positive for producers and injectors.
   There is no unit conversion: convert before you write a file.
3. **Block numbers count from 1** (x fastest, then y, then z, top layer first). Positions in lists inside `--set` paths
   count from 0 (`producers.0` is the first producer, `P1`). Wells are named `P1, P2, ...` and `I1, I2, ...`.
4. **Times must be multiples of `dt`** (observation times, schedule events, `--until`, control periods).
5. **Validate before you launch anything big.** `kusim validate model.json` says `runnable`, estimates the time and warns
   about a `dt` that is likely to oscillate (`checks.findings`, with a `suggested_dt`). The commands that run a model
   refuse a model that is too large or too slow (`too_large`, `too_slow`, `output_too_large`, exit code 2); the answer
   names the flag that raises the limit. Raise a limit only when the user wants that run.
6. **Never trust a number without its diagnostics.** A run can finish with `"ok": true` and still be wrong. Check
   `healthy` (false means: pressure left the PVT table, saturations were clipped, a well flowed backwards or a stop
   condition ended the run) and read `diagnostics` before you use a result.
7. **Reading is strict.** An unknown key is an error, not a silent change of the physics. Fix the input using the error's
   `path` and `did_you_mean`; do not retry the same call.
8. **A good history match is not proof that the permeability field is right** (few wells, many blocks). Check the
   match on data that were not used, and report the fit as "reproduces the history".

## Commands

| Command | Use |
|---|---|
| `kusim template -o model.json` | an example model (small layered waterflood) to start from |
| `kusim convert INPUT.DAT -o model.json` | a legacy input file as a JSON model |
| `kusim schema model` (also `schedule`, `observations`, `results`, `state`, `parameters`, `controls`, `prior`) | JSON Schema of a file; read it instead of guessing keys |
| `kusim validate model.json [--schedule s.json] [--restart state.json]` | check without running |
| `kusim edit model.json --set kx=150 --set producers.0.target=2600 -o new.json` | change values by path (or `--patch file.json`, `--in-place`); writes nothing if the result is invalid |
| `kusim diff a.json b.json` | what differs between two models, by path |
| `kusim run model.json --out dir/ [--schedule s.json] [--until DAY] [--state-out s.json] [--restart s.json] [--strict]` | simulate; `dir/results.json` and `dir/summary.json` |
| `kusim misfit model.json --observations obs.json` | run and compare with data: `objective`, `rms`, `by_vector`, `worst` |
| `kusim gradient model.json --observations obs.json --parameters p.json [--controls c.json] [--check N]` | objective and adjoint gradient (about 3 runs) |
| `kusim ensemble model.json --method esmda --observations obs.json --parameters p.json --prior prior.json` | ensemble history matching (ES-MDA or EnKF); writes `posterior_mean.json` |

## Recipes

**What-if on one model.** `edit` -> `run --out` -> read `summary.json` (final rates, pressures, cumulative volumes per
well; `healthy`). Compare two variants with `diff` and by running both. Do not edit a model in place unless the user
asked: write a new file and keep the old one.

**Run in stages and change a well on the way.** `run --until 200 --state-out s200.json`, then
`run --restart s200.json --schedule s.json` (the same schedule file may be given at every stage; events before the
restart are not applied again). A run in stages equals one run exactly. A restart needs the same grid and wells.

**History matching, the short way.** Put the measurements in `observations.json` (`time`, `vector`, `value`, `sigma`;
vectors such as `WBHP:P1`, `WWCT:P1`, `WOPR:P1`, `FOPT`). Run `misfit` to see where the model is off (`worst`,
`by_vector`). Describe what is uncertain in `parameters.json` (for example `kx` as a log field tied to `ky`, `kz`) and how
in `prior.json` (mean, `std` in ln units, correlation ranges). Run `ensemble --method esmda` (recommended; `enkf` is date
by date). Then run `misfit` on `posterior_mean.json`. Look at `healthy`, `warnings`, `posterior_member_rms` (about 1 means
a fit to within the noise), `posterior_spread` (still large: the data did not constrain it). Use 20 to 40 members; fewer
than 10 triggers a warning. The same `--seed` gives the same answer.

**Optimise well controls.** Describe the control periods in `controls.json` (`well`, `times`, `initial`, `lower`,
`upper`), run `gradient --controls controls.json --check 3` (the adjoint should agree with finite differences to about
1e-6 on the components it reports as stable), step the targets against the gradient with `edit`, and **run the simulator
on every candidate**: the objective of this scheme has kinks and jumps, and an optimiser can sit on one or exploit
backflow. Compare `objective` before and after with `misfit`, and stop when it does not improve.

## Reading a run

`summary.json` (also printed by `run`): `final_time`, `stop_reason`, `wells` (per well: control mode and last-step values),
`diagnostics`, `healthy`. In `diagnostics`: `outside_pvt_range` (pressure left the table: the PVT is extrapolated flat,
so the result is not reliable), `n_saturation_clipped`, `n_backflow_well_steps` (a well flowed the wrong way: a
bottom-hole pressure beyond the block pressure), `stop_reason` (a well stop condition ended the run early).

If `healthy` is false: first look at the controls (a producer held below the PVT range, a rate too high for the cells),
then at `dt`. **IMPES is only conditionally stable**: large rates, tiny cells or a large `dt` make the saturation update
oscillate. Reduce `dt` (with `edit --set dt=...`; the `suggested_dt` of `validate` is a fraction `dt/k`, so existing
times stay on the grid) or the rates before anything else.

## Errors

`{"ok": false, "error": "...", "code": "...", "path": "..."}`. Codes: `unknown_key` (see `allowed`, `did_you_mean`),
`missing_key`, `bad_path` (an `--set` path that does not exist), `bad_value`, `invalid_json` (`line`, `column`),
`invalid_value` (data the model rejects, e.g. a block outside the grid), `unknown_vector` and `bad_time` (observations),
`unknown_well` (controls), `state_mismatch` (a restart state from another grid or wells), `prior_mismatch`,
`no_gradient` (use `--method fd` for scalar parameters), `too_large`, `too_slow`, `output_too_large` (the guardrails above),
`bad_argument`. A schedule that names a well the model does not
have, or an event off the `dt` grid, is exit code 2 as well.

## Limits to tell the user about

- Two phases (oil, water), no gas. Oilfield units only. Rectangular grid, no-flow outer boundary.
- Only ES-MDA and the EnKF for ensembles; the prior is Gaussian in the transformed space, exponential correlation, at most
  3000 blocks for a correlated field.
- The adjoint covers the summary vectors `WBHP, WOPR, WWPR, WWCT, WWIR, FPRP, FOPT, FWPT, FWIT` and field parameters
  (`kx`, `ky`, `kz`, `porosity`) and controls; scalar parameters need `--method fd`.
- Memory is not estimated (the adjoint stores the whole run) and there is no wall-clock timeout: use the timeout of
  whatever starts the process.
- There is no HTTP API or MCP server yet; use the command line or the Python package (`import kusim`).
- A CI workflow for Windows, Linux and macOS is configured, but only Windows has been run so far.

## Where to read more

`docs/json-format.md` (every file, every command) - `docs/units-and-assumptions.md` (units, what the model represents,
the guardrails) - `docs/history-matching.md` (rules and limits of history matching) -
`docs/adjoint.md` (what the gradient does and does not include) - `docs/porting-notes.md` (where the physics differs from
the thesis and the FORTRAN) - `docs/05-equations.md` (the equations).
