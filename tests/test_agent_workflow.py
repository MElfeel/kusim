"""A scripted agent that uses only the command line and JSON files (see AGENTS.md).

It plays a whole session: build a model, make "field data" from a hidden truth, see that the model is off, history match
it with an ensemble, optimise an injection rate with the adjoint, and run a forecast in stages. Every step reads the JSON
answer of a command and decides on that alone. If this test fails after a change, the command line contract has broken.
"""

import json
import subprocess
import sys

import pytest

from kusim.cli import main

DATES = [10.0, 20.0, 30.0]
VECTOR_OF = {"WBHP:I1": ("I1", "bhp", 5.0), "WWCT:P1": ("P1", "water_cut", 0.02), "WOPR:P1": ("P1", "oil_rate", 20.0)}


class Agent:
    """Runs commands in-process the way a shell would: one JSON object out, an exit code."""

    def __init__(self, folder, capsys):
        self.folder, self.capsys = folder, capsys

    def path(self, name):
        return str(self.folder / name)

    def __call__(self, *argv, expect=0):
        code = main([str(a) for a in argv])
        answer = json.loads(self.capsys.readouterr().out)
        assert code == expect, (argv, answer)
        if argv[0] != "schema":                      # `schema` prints the document itself (an error still has "ok": false)
            assert answer["ok"] is (expect != 2), (argv, answer)
        return answer

    def read(self, name):
        return json.loads((self.folder / name).read_text())

    def write(self, name, data):
        (self.folder / name).write_text(json.dumps(data))
        return self.path(name)


@pytest.fixture
def agent(tmp_path, capsys):
    return Agent(tmp_path, capsys)


def test_a_whole_session_through_the_command_line(agent):
    # 1. a model to start from, and a look at what it is
    agent("template", "--nx", 4, "--ny", 3, "--nz", 2, "--days", 30, "-o", agent.path("model.json"))
    info = agent("validate", agent.path("model.json"))
    assert info["n_cells"] == 24 and [w["name"] for w in info["producers"] + info["injectors"]] == ["P1", "I1"]
    model = agent.read("model.json")

    # 2. "field data": a hidden truth (twice the permeability) run once, measurements read from its results
    double = lambda key: json.dumps([2.0 * v for v in model[key]]) if isinstance(model[key], list) else 2.0 * model[key]
    agent("edit", agent.path("model.json"), "--set", f"kx={double('kx')}", "--set", f"ky={double('ky')}",
          "--set", f"kz={double('kz')}", "-o", agent.path("truth.json"))
    agent("run", agent.path("truth.json"), "--fields", "none", "--out", agent.path("truth"))
    rows = {(r["well"], r["time"]): r for r in agent.read("truth/results.json")["wells"]}
    observations = [{"time": t, "vector": v, "value": rows[(well, t)][column], "sigma": sigma}
                    for t in DATES for v, (well, column, sigma) in VECTOR_OF.items()]
    obs = agent.write("observations.json", observations)

    # 3. the model as it is does not reproduce the data
    before = agent("misfit", agent.path("model.json"), "--observations", obs)
    assert before["healthy"] and before["rms"] > 2.0
    assert before["by_vector"]["WBHP:I1"]["rms"] == max(v["rms"] for v in before["by_vector"].values())   # the pressure is off most

    # 4. history match with ES-MDA; the answer says whether to trust it
    params = agent.write("parameters.json", {"parameters": [
        {"type": "field", "field": "kx", "transform": "log", "lower": 1.0, "ties": ["ky", "kz"]}]})
    prior = agent.write("prior.json", {"parameters": {"log_kx": {"std": 0.6, "range_xy": 150.0, "range_z": 20.0}}})
    matched = agent("ensemble", agent.path("model.json"), "--method", "esmda", "--observations", obs, "--parameters", params,
                    "--prior", prior, "--members", 12, "--seed", 3, "--out", agent.path("match"))
    assert matched["healthy"] and matched["metrics"]["posterior_member_rms"] < 0.5 * matched["metrics"]["prior_member_rms"]
    after = agent("misfit", matched["written"]["posterior_mean_model"], "--observations", obs)
    assert after["rms"] < 0.5 * before["rms"]
    changes = agent("diff", agent.path("model.json"), matched["written"]["posterior_mean_model"])
    assert not changes["identical"] and any(c["path"] == "kx" for c in changes["differences"])
    agent("validate", matched["written"]["posterior_mean_model"])                      # what it wrote is a valid model
    best = matched["written"]["posterior_mean_model"]

    # 5. optimise the injection rate with the adjoint: ask for more oil at day 30, step against the gradient, and run
    #    the simulator on every candidate (the objective has kinks: never trust the optimiser alone)
    target = agent("misfit", best, "--observations", obs)["table"]
    wanted = [{"time": 30.0, "vector": "WOPR:P1", "value": 1.15 * next(r["simulated"] for r in target
                                                                       if r["vector"] == "WOPR:P1" and r["time"] == 30.0),
               "sigma": 5.0}]
    goal = agent.write("goal.json", wanted)
    controls = {"controls": [{"well": "I1", "times": [0, 15], "initial": [150.0, 150.0], "mode": "water_rate",
                              "lower": 0.0, "upper": 1000.0}]}
    cfile = agent.write("controls.json", controls)
    g = agent("gradient", best, "--observations", goal, "--controls", cfile, "--check", 2)
    assert g["check"]["n_stable"] >= 1 and g["check"]["max_relative_error"] < 1e-3
    gradient, x = g["variables"]["I1_targets"]["gradient"], g["variables"]["I1_targets"]["x"]

    def objective_of(targets):
        schedule = agent.write("candidate.json", [{"time": t, "well": "I1", "mode": "water_rate", "target": v}
                                                   for t, v in zip([0, 15], targets)])
        result = agent("misfit", best, "--observations", goal, "--schedule", schedule)
        assert result["healthy"]
        return result["objective"]

    start = objective_of(x)
    step, candidate = 0.2 * sum(x) / len(x) / max(abs(v) for v in gradient), None
    for _ in range(8):                                                                  # backtracking line search
        trial = [max(0.0, a - step * b) for a, b in zip(x, gradient)]
        if objective_of(trial) < start:
            candidate = trial
            break
        step /= 2.0
    assert candidate is not None and objective_of(candidate) < start
    assert sum(candidate) > sum(x)                                                      # more injection, more oil

    # 6. forecast in stages with a well change at day 15; the same answer as one run with the schedule
    schedule = agent.write("later.json", [{"time": 15.0, "well": "I1", "target": 80.0}])
    agent("run", best, "--schedule", schedule, "--state-out", agent.path("whole.json"))
    first = agent("run", best, "--schedule", schedule, "--until", 15, "--state-out", agent.path("s15.json"))
    second = agent("run", best, "--schedule", schedule, "--restart", agent.path("s15.json"),
                   "--state-out", agent.path("staged.json"))
    assert first["final_time"] == 15.0 and second["start_time"] == 15.0 and second["final_time"] == 30.0
    assert second["wells"]["I1"]["water_rate"] == 80.0
    assert agent.read("whole.json") == agent.read("staged.json")


def test_an_agent_can_recover_from_its_mistakes(agent):
    """Errors are data: each one says where the input is wrong and what to do."""
    agent("template", "--nx", 3, "--ny", 2, "--nz", 1, "--days", 10, "-o", agent.path("m.json"))
    bad = agent("edit", agent.path("m.json"), "--set", "producers.0.min_bph=1500", "-o", agent.path("n.json"), expect=2)
    assert bad["code"] == "unknown_key" and bad["did_you_mean"] == {"min_bph": "min_bhp"}
    fixed = agent("edit", agent.path("m.json"), "--set", f"producers.0.{next(iter(bad['did_you_mean'].values()))}=1500",
                  "-o", agent.path("n.json"))
    assert fixed["changes"][0]["new"] == 1500
    wrong_time = agent.write("o.json", [{"time": 2.5, "vector": "FOPT", "value": 1.0, "sigma": 1.0}])
    bad = agent("misfit", agent.path("n.json"), "--observations", wrong_time, expect=2)
    assert bad["code"] == "bad_time"
    wrong_name = agent.write("o2.json", [{"time": 5, "vector": "WBHP:P7", "value": 1.0, "sigma": 1.0}])
    bad = agent("misfit", agent.path("n.json"), "--observations", wrong_name, expect=2)
    assert bad["code"] == "unknown_vector" and "WBHP:P1" in bad["allowed"]
    schema = agent("schema", "observations")                                            # or look at the schema first
    assert schema["items"]["required"] == ["sigma", "time", "value", "vector"]


def test_the_console_entry_point_prints_one_json_object(tmp_path):
    """What a non-Python caller sees: a process that prints one JSON object and exits with the documented code."""
    run = lambda *argv: subprocess.run([sys.executable, "-m", "kusim", *argv], capture_output=True, text=True)
    done = run("version")
    assert done.returncode == 0 and json.loads(done.stdout)["ok"] is True
    done = run("validate", str(tmp_path / "missing.json"))
    assert done.returncode == 2 and json.loads(done.stdout)["ok"] is False
    done = run("run", "--solver", "magic", str(tmp_path / "x.json"))
    assert done.returncode == 2 and json.loads(done.stdout)["ok"] is False
