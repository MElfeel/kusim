"""The problems of the agentic tutorial (``tutorial_agentic_workflow.ipynb``): files for an agent to work on, and the answers
that the agent does not see.

``make_workspace(folder)`` writes, in ``folder/work/`` (the agent's workspace):

    model.json          Task A: a small layered waterflood (``kusim template``)
    model_b.json        Task B: the same model with a time step that is too large and a high injection rate
    field/              Task C: what a field team would hand over
        waterflood.DATA, waterflood_props.INC   an ECLIPSE-style deck (``examples/eclipse``)
        history.csv     measured data of days 30-390 (field well names, bar, m3/d, percent, a few blank cells)
        later.csv       measured data of days 420-600 (not used for the calibration: for checking it)
        parameters.json what is uncertain (the permeability field), prior.json what we believe about it

and, in ``folder/hidden/``, the **hidden truth** (the permeability field that produced the field data). The agent has no tool that
can read ``hidden/``; ``score_against_truth`` uses it for the audit at the end of the tutorial.
"""

import contextlib
import dataclasses
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from kusim import load_model, load_schedule, read_deck, save_model, save_schedule
from kusim.cli import main
from kusim.ensemble import correlated_log_field
from kusim.history import run_to

HERE = Path(__file__).resolve().parent
DECK = HERE / "eclipse" / "waterflood.DATA"
LAYER_MEAN_KX = [120.0, 600.0, 250.0]               # md, the layer averages of the deck
PRIOR_STD, RANGE_XY, RANGE_Z, TRUTH_SEED = 0.7, 900.0, 60.0, 7
START = pd.Timestamp("2020-01-01")                    # START of the deck: day 0
FIELD_NAMES = {"P1": "PROD-A", "P2": "PROD-B", "I1": "INJ-1"}
SIGMA_PSI, SIGMA_WC = 10.0, 0.02
HISTORY = [float(t) for t in range(30, 391, 30)]
LATER = [float(t) for t in range(420, 601, 30)]
BAR, M3 = 14.5037738, 6.28981077                       # psi per bar, STB per m3


def _field_csv(path, table, days, rng, blank=()):
    """Write the 'database export': one row per date and well, in the field's names and units."""
    rows = []
    for day in days:
        date = (START + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        row = table.loc[day]
        for kusim_name, field_name in FIELD_NAMES.items():
            if kusim_name.startswith("P"):
                wc = float(np.clip(row[f"WWCT:{kusim_name}"] + SIGMA_WC * rng.standard_normal(), 0.0, 1.0))
                bhp = row[f"WBHP:{kusim_name}"] + SIGMA_PSI * rng.standard_normal()
                rows.append([date, field_name, round(row[f"WOPR:{kusim_name}"] / M3, 1), round(100 * wc, 2), round(bhp / BAR, 2), None])
            else:
                bhp = row["WBHP:I1"] + SIGMA_PSI * rng.standard_normal()
                rows.append([date, field_name, None, None, round(bhp / BAR, 2), round(row["WWIR:I1"] / M3, 1)])
    frame = pd.DataFrame(rows, columns=["Date", "Well", "Oil rate (m3/d)", "Water cut (%)", "BHP (bar)", "Water injection (m3/d)"])
    for index in blank:                                 # gauges that failed
        frame.loc[index, "BHP (bar)"] = None
    frame.to_csv(path, index=False)


def make_workspace(folder, seed=21):
    """Write the task files (see the module docstring); returns ``{"work": Path, "hidden": Path}``."""
    folder = Path(folder)
    work, hidden = folder / "work", folder / "hidden"
    for sub in (work / "field", hidden):
        sub.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        main(["template", "-o", str(work / "model.json")])
    model = load_model(work / "model.json")
    # Task B: a time step of 25 days with a high injection rate: the explicit saturation update will oscillate
    save_model(model.with_changes(dt=25.0, injectors=[dataclasses.replace(model.injectors[0], target=2000.0)]),
               work / "model_b.json")
    # Task C: the deck, the field data from a hidden truth, what is uncertain
    shutil.copy(DECK, work / "field" / DECK.name)
    shutil.copy(DECK.with_name("waterflood_props.INC"), work / "field" / "waterflood_props.INC")
    imported = read_deck(DECK)
    deck_model, schedule = imported.model, imported.schedule
    field = correlated_log_field(deck_model.grid, LAYER_MEAN_KX, PRIOR_STD, RANGE_XY, RANGE_Z, np.random.default_rng(TRUTH_SEED))[:, 0]
    kx = np.exp(field)
    truth = deck_model.with_changes(kx=kx, ky=kx, kz=0.1 * kx, note="the hidden truth")
    save_model(truth, hidden / "truth_model.json")
    run = run_to(truth, HISTORY + LATER, schedule=schedule, vectors=["WBHP:P1", "WBHP:P2", "WBHP:I1", "WWCT:P1", "WWCT:P2", "WOPR:P1",
                                                                      "WOPR:P2", "WWIR:I1"])
    table = run.table.set_index("TIME")
    rng = np.random.default_rng(seed)
    _field_csv(work / "field" / "history.csv", table, HISTORY, rng, blank=(4, 17))
    _field_csv(work / "field" / "later.csv", table, LATER, rng)
    (work / "field" / "parameters.json").write_text(json.dumps({"parameters": [
        {"type": "field", "field": "kx", "transform": "log", "lower": 1.0, "upper": 10000.0, "ties": ["ky", "kz"]}]}, indent=1))
    (work / "field" / "prior.json").write_text(json.dumps({"parameters": {"log_kx": {
        "std": PRIOR_STD, "mean_by_layer": LAYER_MEAN_KX, "range_xy": RANGE_XY, "range_z": RANGE_Z}}}, indent=1))
    return {"work": work, "hidden": hidden}


def score_against_truth(model_path, hidden):
    """How far a calibrated model's permeability is from the hidden truth: the RMS error of ln(kx), and the same for the deck's
    model (the starting point). Smaller is better; a calibrated model that is not better than the start was not helped by the data."""
    truth = load_model(Path(hidden) / "truth_model.json")
    start = read_deck(DECK).model
    rms = lambda m: float(np.sqrt(np.mean((np.log(m.kx) - np.log(truth.kx)) ** 2)))
    return {"deck_model": round(rms(start), 3), "calibrated": round(rms(load_model(model_path)), 3)}
