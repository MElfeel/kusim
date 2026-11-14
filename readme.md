# KUSIM

A 3-D, two-phase (oil/water) reservoir simulator in Python. It is a port of a legacy FORTRAN IMPES code, written to be
read: variable names follow the equations of the original thesis, and every place where the Python differs from the
FORTRAN is recorded in [`docs/porting-notes.md`](docs/porting-notes.md).

You can use it in three ways:

- as a **Python library** (build a model, run it, look at the results);
- from the **command line with JSON files** (for scripts and LLM agents; one JSON object printed per command);
- as the basis for **history matching and optimisation** (ES-MDA, EnKF, and adjoint gradients).

It is a teaching and research code, not a validated commercial simulator (see [Limits](#limits)).

## Citing

If you use kusim in research, please cite it. GitHub's "Cite this repository" button (right-hand side of the repository
page) reads [`CITATION.cff`](CITATION.cff) and gives you BibTeX or APA. The archived releases are on Zenodo; the DOI [10.5281/zenodo.24144332](https://doi.org/10.5281/zenodo.24144332)
always points to the latest one, and each release also has its own DOI (version 0.1.0: 10.5281/zenodo.24144333).

## Install (step by step)

New to the terminal? Follow the longer [setup guide for complete beginners](docs/setup-for-beginners.md).

You need **git** and **uv**. You do not need to install Python yourself (uv downloads Python 3.12 for you) and
you do not need to know any reservoir simulation to start.

**1. Install the tools** (once)

| | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| git | <https://git-scm.com/downloads> | usually installed; otherwise <https://git-scm.com/downloads> |
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

Close and reopen the terminal, then check with `git --version` and `uv --version`.

**2. Get the code**

```
git clone https://github.com/MElfeel/kusim.git
cd kusim
```

**3. Install kusim and its packages**

```
uv sync --extra dev
```

This creates a private `.venv` folder in `kusim/` (so kusim does not disturb other Python projects), installs Python if
needed, and installs kusim in editable mode (changes you make to the code in `src/kusim/` take effect at once) with the
exact package versions in `uv.lock`, plus pytest and plotly. A lighter install, without the tests, is `uv sync`.
Further extras: `--extra api` (the HTTP server), `--extra excel` (Excel history files).

There is nothing to activate: put `uv run` in front of a command and uv runs it inside `.venv`
(for example `uv run kusim version` or `uv run python examples/01_waterflood_1d.py`).

**4. Start JupyterLab**

```
uv run --with jupyterlab jupyter lab
```

A browser tab opens. Use the file browser on the left to open a notebook, for example
`course/day01/day01_simulation_models.ipynb`. Stop JupyterLab with `Ctrl+C` in the terminal. (`--with jupyterlab` adds
JupyterLab to that one run, so it does not change the project files.) In a notebook cell, `import kusim` should run
without an error.

**5. Check that it works**

```
uv run kusim version
uv run pytest -q -m "not slow"        # optional: about 45 s, all tests should pass
```

Below, commands are written as `kusim ...` and `python ...`. Write `uv run` in front of them, or activate the
environment once per terminal (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on macOS / Linux),
after which the commands work as written. `python -m kusim ...` is the same program as `kusim ...`.

## Your first simulation (5 minutes)

**A. From the command line.** Write an example model, check it, run it:

```
kusim template -o model.json            # a small layered waterflood: 10 x 10 x 3 blocks, one injector, one producer
kusim validate model.json               # checks the model without running it
kusim run model.json --out results/     # writes results/results.json and results/summary.json
```

`results/summary.json` says whether the run was healthy and gives the final rates and cumulative volumes of each well.

**B. From Python.** Run the same model and plot the water cut of the producer:

```python
import matplotlib.pyplot as plt
import kusim

model = kusim.load_model("model.json")
results = kusim.Simulator(model).run()

p1 = results.well("P1")                  # a pandas table: one row per time step
print(p1[["time", "oil_rate", "water_rate", "water_cut"]].tail())

plt.plot(p1["time"], p1["water_cut"])
plt.xlabel("time (days)")
plt.ylabel("water cut of P1")
plt.show()
```

**C. Change something and compare.** Raise the injection rate of the injector `I1` from 150 to 800 STB/day and compare the two runs:

```
kusim describe model.json                                    # what is in the model, with the name of each well
kusim edit model.json --set I1.target=800 -o fast.json       # wells are addressed by name
kusim run fast.json --out results_fast/
kusim compare results results_fast                           # well by well
```

The comparison lists, for each well, the rates, pressures and cumulative volumes of both runs and their difference. Try
plotting the water cut of `P1` for both runs with the snippet of step B.

**D. Open the tutorial notebook.** It builds a model from scratch, step by step, and shows line plots, layer maps, 3-D
views and movies:

```
uv run --with jupyterlab jupyter lab examples/tutorial_first_simulation.ipynb
```

## Where to go next

| I want to... | Go to |
|---|---|
| learn by building a model in a notebook | `examples/tutorial_first_simulation.ipynb` |
| let an LLM agent drive kusim | `examples/tutorial_agentic_workflow_local.ipynb` (free, local models with Ollama), `tutorial_agentic_workflow_frontier.ipynb` (Anthropic / OpenAI), `tutorial_agentic_workflow.ipynb` (provider-neutral); setup in [`docs/agent-tutorial.md`](docs/agent-tutorial.md) |
| see the simulator checked against theory | `python examples/01_waterflood_1d.py` (1-D waterflood vs Buckley-Leverett, writes a PNG) |
| see all the viewers (maps, GIFs, interactive 3-D, ParaView files) | `python examples/03_visualization.py` (about 25 s, writes to `examples/output/`) |
| history match | `python examples/07_esmda_twin_experiment.py`, then [`docs/history-matching.md`](docs/history-matching.md) |
| read the thesis case | `python examples/02_legacy_case.py`, [`docs/06-applications.md`](docs/06-applications.md) |
| use kusim from a program or agent | [`AGENTS.md`](AGENTS.md), [`docs/json-format.md`](docs/json-format.md) |

Run the scripts from the repository root (the folder that contains `pyproject.toml`).

## User guide

| Topic | Document |
|---|---|
| Inputs, outputs and every command of the command line (JSON format) | [`docs/json-format.md`](docs/json-format.md) |
| Units, assumptions and guardrails | [`docs/units-and-assumptions.md`](docs/units-and-assumptions.md) |
| The equations (thesis chapter 5, with the printed typos) and the applications (chapter 6) | [`docs/05-equations.md`](docs/05-equations.md), [`docs/06-applications.md`](docs/06-applications.md) |
| Where the Python differs from the FORTRAN or the thesis | [`docs/porting-notes.md`](docs/porting-notes.md) |
| History matching (ES-MDA, EnKF) | [`docs/history-matching.md`](docs/history-matching.md) |
| Plotting from JSON descriptions (`kusim tables`, `plot-template`, `plot`) | [`docs/plotting.md`](docs/plotting.md) |
| ECLIPSE-style decks (`import-deck`) | [`docs/eclipse-deck.md`](docs/eclipse-deck.md) |
| Excel / CSV history (`import-history`) | [`docs/excel-history.md`](docs/excel-history.md) |
| HTTP API (`kusim serve`) | [`docs/http-api.md`](docs/http-api.md) |

A legacy `INPUT.DAT` can be converted with `kusim convert INPUT.DAT -o model.json`; the thesis input is in
`legacy/fortran-3d3/INPUT.DAT`.

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

## Repository layout

| Path | Contents |
|---|---|
| `src/kusim/` | the package: grid, fluids, wells, model, simulator, solver, results, JSON layer (`spec.py`), CLI (`cli.py`), history matching (`parameters.py`, `schedule.py`, `history.py`, `ensemble.py`), adjoint (`adjoint.py`, `gradients.py`), plotting (`viz/`, `figure.py`) |
| `examples/` | tutorial notebooks and runnable scripts (see above) |
| `docs/` | the user guide and the equations |
| `tests/` | pytest suite (`test_agent_workflow.py` is a scripted agent session that uses only the command line) |
| `AGENTS.md` | guide for agents and scripts that use the command line |
| `legacy/fortran-3d3/INPUT.DAT` | the thesis input file used by a test and by `examples/02_legacy_case.py` |

## Tests

```
uv run pytest -q -m "not slow"        # fast suite, about 45 s
uv run pytest -q                      # everything, about 2.5 min
```

The continuous-integration workflow (`.github/workflows/tests.yml`) runs the fast suite on Linux, Windows and macOS
for each push and pull request, and the full suite every night.

## Limits

- Oilfield units only; no gas phase; no unit conversion in the JSON format.
- The original FORTRAN executable and its output files could not be used as a reference. The simulator is validated
  against Buckley-Leverett, material balance and the first-step well rates of the legacy input
  (see [`docs/units-and-assumptions.md`](docs/units-and-assumptions.md)).
- IMPES is only conditionally stable; `kusim validate` and `kusim run` warn when a model looks unstable.
- History matching recovers the permeability field only partly; the fit to the data improves more reliably than the field
  (see [`docs/history-matching.md`](docs/history-matching.md)).
- The HTTP API (`kusim serve`) has no TLS or user accounts: keep it on localhost or behind a reverse proxy. There is no
  MCP server.

## License

See [`LICENSE`](LICENSE).
