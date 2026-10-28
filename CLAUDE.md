# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A teaching project: port a legacy FORTRAN 3-D two-phase (oil/water) IMPES reservoir simulator to Python modules and Jupyter notebooks, and build a course around it for undergraduate petroleum engineers who are learning reservoir simulation and Python at the same time. `docs/original-task-brief.md` is the original task brief (7 steps); `readme.md` is the user README; `PLAN.md` is the work plan and records findings; `docs/porting-notes.md` records every place where the Python deliberately differs from the FORTRAN or the thesis.

**Guiding principle: readability for students first.** Prefer clear, well-named code that maps visibly to the thesis equations over cleverness. Where the thesis chapters and the FORTRAN disagree, keep the thesis physics and the code's numerical scheme, fix clear mistakes, and write the decision in `docs/porting-notes.md`.

Programs and LLM agents that *use* kusim (not develop it) are pointed to `AGENTS.md`; keep it true when the command line changes (`tests/test_agent_workflow.py` plays that session).

## Commands

```
py -m pip install -e .[dev]        # install the package (editable), pytest, plotly (interactive 3-D) and kaleido (static export of plotly)
py -m pytest -q                    # all tests (about 2.5 min: the ensemble-filter and adjoint-workflow tests are marked `slow`)
py -m pytest -q -m "not slow"      # the fast tests (about 45 s)
py -m pytest tests/test_simulator.py -k material_balance   # one test or group
py -m kusim template -o model.json   # then: py -m kusim run model.json --out results/  (JSON in, JSON out; see docs/json-format.md)
py -m kusim edit model.json --set producers.0.target=2600 -o new.json   # also: diff, schema, run --until/--restart/--state-out, misfit/gradient/ensemble (see docs/json-format.md)
py examples/01_waterflood_1d.py    # 1-D waterflood vs Buckley-Leverett, writes a PNG next to it
py examples/02_legacy_case.py      # thesis Application 1 (INPUT.DAT), writes a PNG
py examples/03_visualization.py    # every viewer: PNGs, GIFs, interactive HTML, ParaView files -> examples/output/ (about 25 s)
py examples/07_esmda_twin_experiment.py  # ES-MDA twin experiment (the history-matching method of the course, Week 5), writes a PNG (about 10 s)
py examples/04_enkf_twin_experiment.py   # Ensemble Kalman Filter twin experiment, kept for reference (not used in the course), writes a PNG (a few seconds)
py examples/05_adjoint_history_matching.py        # history matching with adjoint gradients on the same twin, writes a PNG (about 50 s)
py examples/06_adjoint_production_optimisation.py # fixed-volume injection allocation with adjoint gradients, writes a PNG (about 30 s)
```

On this Windows machine use `py` (the `python` command is only the Store stub). There is no Fortran compiler and no pandoc.

## Layout

- `src/kusim/` is the package (see below). `tests/` are pytest tests. `examples/` are runnable scripts.
- `docs/05-equations.md` and `docs/06-applications.md` are transcriptions of thesis Chapters 5 and 6 (with LaTeX), including a table of printed typos at the end of chapter 5. The original PDFs, Word files and page images are in `legacy/thesis/`.
- `legacy/` holds everything inherited (see `legacy/README.md`). `legacy/fortran-3d3/` is the **reference FORTRAN** (`3DTP(11).FOR`, reads one `INPUT.DAT`). `legacy/fortran-kusim/` is a later, different variant (hard-coded initial `PCOW`, separate `SYSTEM/PVT/SAT.DAT`); ignore it.
- **Do not trust the shipped `3DTP(11).exe` or `*.OUT` files in `legacy/fortran-3d3/`.** The exe was built from a different source (it opens `SYSTEM.DAT` etc.) and crashes on every input tried; the `.OUT` files cannot be reproduced (`NP.OUT` is zeros, `PO.OUT` is corrupted). Only the first step's well rates are comparable (a test uses them). See `PLAN.md` and `legacy/README.md`.
- `python_book/` holds Severance's *Python for Informatics* PDF (Python 2 era) for course inspiration; not yet converted.
- `history_matching_lab/` has notebooks to explore and compare ES-MDA, the EnKF and the adjoint on one twin experiment (`lab.py` holds the shared helpers; not part of the course). `eclipse_comparison/` is the ECLIPSE check (see its README).

## Architecture of `src/kusim/`

Data flows `ReservoirModel` -> `Simulator` -> `Results`. Everything is flat 1-D NumPy arrays of length `n_cells`, ordered x fastest, then y, then z (top layer first), matching the thesis block numbers (user-facing block numbers start at 1; array indices at 0).

- `grid.py` `Grid`: block sizes, volumes, cell-centre depths, and for each of six directions a neighbour index array plus an "is there a neighbour" mask. A closed face gets zero transmissibility, which gives the no-flow boundary.
- `tables.py`, `fluids.py`: piecewise-linear lookup (value and slope, flat extrapolation) and the PVT / rock-fluid tables. The code stores **1/B** (not B) because that is what the equations use.
- `wells.py`: `Producer`/`Injector` dataclasses (per-well control mode, limits, one or several completion blocks, wellbore gravity) and the Peaceman well index.
- `model.py`: `ReservoirModel`, `Equilibrium` (capillary-gravity initial state) and `UniformStart`.
- `simulator.py`: the IMPES step, in the order of thesis section 5.8 (properties, upstream-weighted transmissibilities, upstream block by phase potential: `_neighbour_is_upstream`, storage coefficients, gravity and capillary terms, pressure equation with well terms, solve, well rates, explicit saturation update). Variable names follow the thesis (`lam_o`, `t_o`, `cpoo`, `alpha`, ...); the module docstring has a glossary.
- `parameters.py`, `schedule.py`, `history.py` (+ `ReservoirModel.with_changes`, `Simulator.get_state/set_state/set_well_control/well_data/diagnostics`): the interface for history matching, ensemble methods and agents: edit a model as a copy (`with_changes`) or as a parameter vector (`ParameterSet`: log permeability, ties for `ky`/`kz`, bounds), restart from a saved `SimState`, time-varying well controls, ECLIPSE-named summary vectors, `Observations`, `run_ensemble` (optionally parallel), `misfit`, run diagnostics. `ensemble.py` has the two ensemble methods on this interface: ES-MDA (the method taught in Week 5; no localisation) and an EnKF (with Gaspari-Cohn localisation), a Gaussian prior sampler, and `run_esmda`/`run_enkf` (what `kusim ensemble` runs). `examples/esmda.py` and `enkf_reference.py` re-export their functions for the examples and notebooks, `esmda_twin.py` and `enkf_twin.py` are the twin experiments. Guide, rules and limits: `docs/history-matching.md`.
- `adjoint.py`, `gradients.py` (+ `ControlParameter`/`ControlSet` in `schedule.py`, `ParameterSet.pullback`, `linear_terms` in `history.py`): the **adjoint** of the IMPES step, written by hand as a separate functional copy of `Simulator.step` (`forward_step`, and its transpose `backward_step`), and `HistoryProblem`, which gives the objective and its gradient (permeability, porosity, well controls) for any optimiser at the cost of about 3 simulations. The copy reproduces the simulator exactly (tests keep it so). Guide, evidence and limits: `docs/adjoint.md`; workflows in `examples/05` and `06` (`adjoint_workflows.py`).
- `spec.py`, `cli.py` (`kusim` command, `python -m kusim`): the model, schedule, observations and results as plain **JSON**, strict on reading (unknown or missing keys raise `ValueError` naming them), and a command line that prints one JSON object per command. `ReservoirModel.to_dict/from_dict/to_json/from_json` and `Results.to_dict/to_json/from_json` delegate to `spec.py`. If you add a field to `ReservoirModel`, wells or `Results`, add it to `spec.py` too (`tests/test_json_cli.py` round-trips the models and fails otherwise). `SimState` and `ParameterSet` have `to_dict/from_dict/to_json` too (restart state; `kusim run --restart`), `edit.py` (edit a model by path, diff two models) and `schema.py` (JSON Schema, kept in step with `spec.py` by `tests/test_edit_schema.py`: if you add a key to `spec.py`, add it to the schema). Reader errors are `SpecError` (a `ValueError` with `path`, `code`, hints) and the CLI prints those fields. `checks.py` (`check_model`: size, estimated time, output size and a stability heuristic from `Simulator.well_block_throughput`; the commands that run a model call it first). Format and limits: `docs/json-format.md`; units, assumptions and the guardrails: `docs/units-and-assumptions.md`.
- `solver.py`: sparse direct solver (default) and a plain-loop Gaussian elimination kept for teaching.
- `io.py`: `read_legacy_input` replicates FORTRAN list-directed reading of `INPUT.DAT` (positional, labels ignored, a bare `READ` consumes one line).
- `analytical.py`: Buckley-Leverett (Welge) solution used to validate the simulator.
- `results.py`: `Results` with field snapshots and a pandas well-history table.
- `cases.py`: ready-made models (`layered_waterflood`, Corey relperm and a simple PVT table) for examples and tests.
- `viz/`: `maps.py` (layer maps, cross-sections, matplotlib 3-D voxels), `wells.py` (well dashboard; `plot_inflow_profile` and `plot_perforation_rates` for multi-layer wells), `animate.py` (GIFs via Pillow), `interactive.py` (Plotly 3-D with time slider, only the visible faces are drawn; plotly is optional and imported lazily), `vtk.py` (pure-Python `.vtu` + `.pvd` export for ParaView plus `_wells.vtp` wellbores; depth is written as elevation = -depth). `fields.py` holds the shared helpers (fixed colour limits over a run, snapshot picking, cut-away mask). `Grid` provides the plotting geometry (`x_edges`, `y_edges`, `z_edges`, `cell_corners`, `cell_centres`).

## Things that will bite you

- **Upstream weighting is by potential, not pressure (porting-notes decision 18, a deliberate deviation from the thesis and the FORTRAN).** `Simulator(model, upstream="pressure")` gives the thesis rule back; it traps sinking water in vertical flow. Results of models with vertical flow and a density contrast differ between the two after water breakthrough; horizontal 1-D results do not.
- Units are oilfield units; rates are STB/day, entered positive for both producers and injectors. Well terms carry a 5.615 ft³/bbl factor that the FORTRAN omitted (porting-notes decision 1).
- IMPES is only conditionally stable: with large rates, tiny cells or large `dt` the explicit saturation update oscillates (the thesis's own `INPUT.DAT` at `dt = 1` does). Test models should use realistic rates and volumes; `tests/conftest.py::make_model` does.
- A well can be completed in several blocks (`Producer(block=grid.column_blocks(i, j), ...)`). Rate-controlled wells share the rate by well index x mobility (explicit, no cross-flow); pressure-controlled wells flow per perforation. Wellbore pressure head is included (details in porting-notes decisions 16-17); results carry per-perforation rates in `Results.perforations`.
- **History matching** (`docs/history-matching.md`): models are copied, never edited in place; schedule events, report times and restart states must be multiples of `dt`; a `SimState` carries well controls (a producer that reached `min_bhp` is on "bhp") and cumulative volumes, not only p and Sw; `set_state` clips Sw to `[swc, 1 - sor]` (count in `RunRecord.adjusted`) but not pressure; check `RunRecord.ok` / `Simulator.diagnostics()` (PVT range, Sw clipping) before trusting a member's misfit; one `ProcessPoolExecutor` per loop on Windows. The twin experiment recovers the permeability field only partly (data fit improves on every seed, the field on about two thirds, for the EnKF and for ES-MDA alike): do not read more into it than that. Do not add distance localisation to ES-MDA by weighting only the state-data covariance: it made the update overshoot (NaN from pass 3) at every length tried, 500 to 2000 ft.
- **Adjoint gradients** (`docs/adjoint.md`): the objective of this explicit upwind scheme has kinks at tiny scales and **jumps** (the upstream choice uses the old state; `min_bhp` is applied as a switch after the step), and an optimiser will sit on a jump or exploit backflow (a bottom-hole pressure above the block pressure makes a producer inject). Always run the simulator again on an optimised result; check gradients with `HistoryProblem.check` (steps 1e-4 to 1e-7, stable estimates only); read `diagnostics` (`mode_switches`, `backflow_well_steps`, ...). If you change `Simulator.step` or the wells, change `adjoint.py` too: the tests that compare the copy with the simulator will fail until you do.
- Visualization output is easy to get subtly wrong without looking at it: after changing `viz/`, run `examples/03_visualization.py` and view the PNGs / a GIF frame; `tests/test_viz.py` only checks structure (and VTK corner handedness). The interactive HTML has not been opened in a real browser, only exported to PNG with kaleido.
- The printed thesis equations contain typos (listed at the end of `docs/05-equations.md`); when in doubt the FORTRAN decides, except where `docs/porting-notes.md` records a deliberate deviation (for example decision 18, upstream by potential).
