"""The three tasks of the agentic tutorial, and a scripted plan for each (what a careful agent would do).

``TASKS[name]`` is the request given to the agent. ``SCRIPTS[name]`` is a plan for ``agent_lab.ScriptedLLM``: the same tool calls
a good model makes, so that the tutorial runs without a key and gives the same trace every time. Final answers are written from
the numbers in the tool answers (``results(messages)``), not typed in advance. ``SCRIPTS["C"]`` ends with an **overclaim** on
purpose: the audit section of the tutorial takes it apart.
"""

import json

TASKS = {
    "A": ("The model is model.json. What happens to the cumulative oil of producer P1 at the end of the run if the producer's "
          "bottom-hole pressure is lowered from 2500 to 2000 psi? Keep model.json as it is; run both cases and compare."),
    "B": ("Run model_b.json and tell me the final oil rate and water cut of P1. I want numbers I can trust: if anything is "
          "wrong with the run or the model, fix it in a copy and say what you changed."),
    "C": ("The folder field/ has what the field team sent: the deck waterflood.DATA, history.csv (measured data, day 0 is the "
          "deck's START, 1 January 2020) and later.csv (data after the history). Import the deck and the history, check how "
          "well the deck model reproduces the history, calibrate the permeability with ES-MDA (parameters.json and prior.json "
          "are in field/), and report. Use the later data to check the calibrated model. Write your conclusion in report.md."),
}


def results(messages):
    """The tool answers so far, as dicts (the newest last)."""
    out = []
    for message in messages:
        if message["role"] == "tool":
            try:
                out.append(json.loads(message["content"]))
            except ValueError:
                pass
    return out


def _runs(messages):
    return [r for r in results(messages) if "wells" in r and "final_time" in r]


def _call(name, **arguments):
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def _final_a(messages):
    runs = _runs(messages)
    base, low = (r["wells"]["P1"]["cum_oil"] for r in runs[-2:])
    return {"text": f"Cumulative oil of P1 at day {runs[-1]['final_time']:g}: {base:,.0f} STB at 2500 psi, {low:,.0f} STB at 2000 psi "
                    f"({100 * (low - base) / base:+.1f} %). Both runs were healthy: {runs[-2]['healthy']} and {runs[-1]['healthy']}. "
                    "Not checked: whether 2000 psi is above the bubble point or the pump limit (kusim has no gas)."}


SCRIPT_A = [
    {"text": "I will validate the model first.", **_call("validate", model="model.json")},
    _call("run", model="model.json", out="runs/base"),
    {"text": "Now the same model with the producer at 2000 psi, written as a new file.",
     **_call("edit", model="model.json", set=["producers.0.target=2000"], output="model_2000.json")},
    _call("validate", model="model_2000.json"),
    _call("run", model="model_2000.json", out="runs/bhp2000"),
    _final_a,
]


def _final_b(messages):
    runs = _runs(messages)
    bad, run = runs[0], runs[-1]
    p1 = run["wells"]["P1"]
    return {"text": f"The model as given was not trustworthy: dt = 25 days made the saturation update oscillate (healthy = {bad['healthy']}). "
                    f"In a copy (model_b_fixed.json) I set dt to 0.78125 days, the value validate suggested; the run is healthy = {run['healthy']}. "
                    f"P1 at day {run['final_time']:g}: oil rate {p1['oil_rate']:.1f} STB/d, water cut {p1['water_cut']:.3f}. "
                    "Not checked: whether a smaller dt changes the answer further (try halving it)."}


SCRIPT_B = [
    {"text": "I run the model as it is first.", **_call("run", model="model_b.json", out="runs/b_asgiven")},
    {"text": "The run is flagged. I ask validate why.", **_call("validate", model="model_b.json")},
    {"text": "I write the fix as a new file.", **_call("edit", model="model_b.json", set="dt=0.78125", output="model_b_fixed.json")},
    {"text": "That call was refused before it ran (set must be a list); corrected.",
     **_call("edit", model="model_b.json", set=["dt=0.78125"], output="model_b_fixed.json")},
    _call("validate", model="model_b_fixed.json"),
    _call("run", model="model_b_fixed.json", out="runs/b_fixed"),
    _final_b,
]


def _misfits(messages):
    return [r["rms"] for r in results(messages) if "rms" in r][-4:]      # deck/history, deck/later, calibrated/history, calibrated/later


def _report_c(messages):
    before_h, before_l, after_h, after_l = _misfits(messages)
    text = (f"# Calibration report\n\nMisfit (RMS of residual/sigma) on the history: {before_h:.2f} before, {after_h:.2f} after ES-MDA.\n"
            f"On the later data: {before_l:.2f} before, {after_l:.2f} after.\n\n"
            "Conclusion: the model is calibrated and the permeability field is now right.\n")
    return {"text": "I write the report.", **_call("write_file", path="report.md", content=text)}


def _final_c(messages):
    before_h, before_l, after_h, after_l = _misfits(messages)
    return {"text": f"Done. The calibrated model reproduces the history (misfit {before_h:.1f} -> {after_h:.2f}) and the later data "
                    f"({before_l:.1f} -> {after_l:.2f}); I conclude that the calibrated permeability is right. See report.md."}


_SIGMA = ["WBHP=10", "WWCT=0.02"]
_MAP = ["PROD-A=P1", "PROD-B=P2", "INJ-1=I1"]
_IMPORT = dict(model="model_c.json", start_date="2020-01-01", every=30, vectors="WBHP,WWCT", sigma=_SIGMA)
SCRIPT_C = [
    {"text": "First, what is in field/?", **_call("list_files", folder="field")},
    _call("import_deck", deck="field/waterflood.DATA", output="model_c.json", schedule_out="schedule_c.json", wells_out="wells_c.json"),
    {"text": "Now the measured history.", **_call("import_history", input="field/history.csv", output="obs_history.json", **_IMPORT)},
    {"text": "The well names in the file are the field's: I map them to kusim's.",
     **_call("import_history", input="field/history.csv", output="obs_history.json", well_map=_MAP, **_IMPORT)},
    _call("import_history", input="field/later.csv", output="obs_later.json", well_map=_MAP, **_IMPORT),
    _call("misfit", model="model_c.json", observations="obs_history.json", schedule="schedule_c.json"),
    _call("misfit", model="model_c.json", observations="obs_later.json", schedule="schedule_c.json"),
    {"text": "Calibration with ES-MDA, 40 members.",
     **_call("ensemble", model="model_c.json", method="esmda", observations="obs_history.json", parameters="field/parameters.json",
             prior="field/prior.json", schedule="schedule_c.json", members=40, seed=11, workers=4, out="calibration")},
    _call("misfit", model="calibration/posterior_mean.json", observations="obs_history.json", schedule="schedule_c.json"),
    _call("misfit", model="calibration/posterior_mean.json", observations="obs_later.json", schedule="schedule_c.json"),
    _report_c,
    _final_c,
]

SCRIPTS = {"A": SCRIPT_A, "B": SCRIPT_B, "C": SCRIPT_C}
