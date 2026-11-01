# KUSIM

A 3-D, two-phase (oil/water) reservoir simulator in Python, ported from a legacy FORTRAN IMPES code and written to be
read: variable names follow the thesis equations, and every place where the Python differs from the FORTRAN is
recorded in [`docs/porting-notes.md`](docs/porting-notes.md).

It can be used three ways: as a Python library, from the command line with JSON files (for scripts and LLM agents),
and as the basis for history matching and optimisation (ensemble methods and adjoint gradients).

## Install

Python 3.10 or newer.

```
pip install -e .            # the simulator (numpy, scipy, pandas, matplotlib)
pip install -e .[api]       # plus FastAPI and uvicorn: kusim serve
pip install -e .[dev]       # plus pytest, plotly, kaleido, jsonschema and what the API tests need
```

## Quick start

From the command line (JSON in, JSON out, see [`docs/json-format.md`](docs/json-format.md)):

```
kusim template -o model.json           # a small layered waterflood to start from
kusim validate model.json
kusim run model.json --out results/    # writes results/results.json and summary.json
```

From Python:

```python
import kusim

model = kusim.load_model("model.json")
results = kusim.Simulator(model).run()
print(results.well("P1").head())       # pandas table: rates, pressures, water cut per time step
```

A legacy `INPUT.DAT` can be converted with `kusim convert INPUT.DAT -o model.json`.

For programs and LLM agents: [`AGENTS.md`](AGENTS.md) (what to run, what to check, what not to trust). Besides `run`, the
command line can `edit` and `diff` models, `misfit` them against data, give adjoint `gradient`s, and history match with an
ensemble (`ensemble`, ES-MDA or EnKF). Input comes from JSON, from an ECLIPSE-style deck (`import-deck`, the supported subset: [docs/eclipse-deck.md](docs/eclipse-deck.md)) and historical data from Excel or CSV (`import-history`, `pip install kusim[excel]`: [docs/excel-history.md](docs/excel-history.md)). `kusim serve` offers the same commands over HTTP (`pip install kusim[api]`, [docs/http-api.md](docs/http-api.md)).

## What the model is

- Black-oil style **two-phase flow** (oil and water, no gas), slightly compressible rock, tabulated PVT and
  relative permeability / capillary pressure.
- **IMPES**: implicit pressure, explicit saturation, upstream-weighted transmissibilities (weighted by phase
  potential, not pressure; porting-notes decision 18). Conditionally stable: large rates, small cells or a large
  time step make the saturation update oscillate. Check `healthy` in the run summary.
- Rectangular grid, no-flow outer boundary, gravity and capillary pressure, equilibrium or uniform initial state.
- Wells: rate or bottom-hole-pressure controlled, Peaceman well index, completions in several layers, time-varying
  controls through a schedule.
- **Units are oilfield units**: ft, md, psi, cp, STB/day, days, 1/psi. Rates are positive for producers and
  injectors. Block numbers in files start at 1 (x fastest, then y, then z, top layer first).

## Layout

| Path | Contents |
|---|---|
| `src/kusim/` | the package: grid, fluids, wells, model, simulator, solver, results, JSON layer (`spec.py`), CLI (`cli.py`), history matching (`parameters.py`, `schedule.py`, `history.py`), adjoint (`adjoint.py`, `gradients.py`), plotting (`viz/`) |
| `AGENTS.md` | guide for agents and scripts that use the command line |
| `tests/` | pytest suite (`test_agent_workflow.py` is a scripted agent session that uses only the command line) |
| `examples/` | a beginner tutorial notebook (`tutorial_first_simulation.ipynb`: build, run, view) and three versions of one on LLM agents driving kusim (`tutorial_agentic_workflow_local.ipynb` with Ollama, `..._frontier.ipynb` with Anthropic or OpenAI, `tutorial_agentic_workflow.ipynb` provider-neutral; `agent_lab.py`) and runnable scripts: waterflood vs Buckley-Leverett, the thesis case, visualisation, ES-MDA, adjoint workflows |
| `docs/` | [equations](docs/05-equations.md), [applications](docs/06-applications.md), [porting notes](docs/porting-notes.md), [units and assumptions](docs/units-and-assumptions.md), [HTTP API](docs/http-api.md), [ECLIPSE decks](docs/eclipse-deck.md), [Excel history](docs/excel-history.md), [JSON format and CLI](docs/json-format.md), [history matching](docs/history-matching.md), [adjoint](docs/adjoint.md) |
| `legacy/fortran-3d3/INPUT.DAT` | the thesis input file used by a test and by `examples/02_legacy_case.py` |

## Tests

```
pytest -q -m "not slow"     # fast suite, about 45 s
pytest -q                   # everything, about 2.5 min
```

The continuous-integration workflow (`.github/workflows/tests.yml`) runs the fast suite on Linux, Windows and macOS
for each push and pull request, and the full suite every night.

## Limits

- Oilfield units only; no gas phase; no unit conversion in the JSON format.
- The shipped FORTRAN executable and `.OUT` files are not a valid reference (see `legacy/README.md`); the simulator is
  validated against Buckley-Leverett, material balance and the first-step well rates of the legacy input.
- The HTTP API (`kusim serve`, `docs/http-api.md`) has no TLS or user accounts: keep it on localhost or behind a reverse proxy. There is no MCP server.
