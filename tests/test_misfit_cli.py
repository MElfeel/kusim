"""kusim misfit: run a model and compare it with observations from the command line."""

import json

import numpy as np
import pytest

from kusim import Simulator
from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.history import Observations, misfit, objective, rms, run_to, summary_vectors, vector_names
from kusim.spec import save_model, save_observations


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def files(tmp_path):
    model = layered_waterflood(nx=4, ny=3, nz=2, days=20.0)
    save_model(model, tmp_path / "m.json")
    obs = Observations([
        {"time": 10.0, "vector": "WBHP:P1", "value": 2400.0, "sigma": 20.0},
        {"time": 20.0, "vector": "WWCT:P1", "value": 0.1, "sigma": 0.05},
        {"time": 20.0, "vector": "FOPT", "value": 1.0, "sigma": 1000.0, "weight": 2.0},
    ])
    save_observations(obs, tmp_path / "o.json")
    return model, obs, tmp_path


def test_vector_names_match_the_simulator():
    model = layered_waterflood(nx=3, ny=2, nz=2, days=3.0)
    sim = Simulator(model)
    sim.step()
    assert sorted(vector_names(model)) == sorted(summary_vectors(sim))


def test_misfit_equals_the_python_calculation(files, capsys):
    model, obs, tmp = files
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "o.json")
    table = misfit(obs, run_to(model, obs.times, vectors=obs.vectors).table)
    assert code == 0 and out["ok"] and out["healthy"] and out["n_missing"] == 0
    assert out["objective"] == pytest.approx(objective(table)) and out["rms"] == pytest.approx(rms(table))
    assert out["by_vector"]["WBHP:P1"]["rms"] == pytest.approx(5.0)       # (100 psi / 20 psi): the producer holds 2500 psi
    assert out["worst"][0]["vector"] == "WBHP:P1"
    assert len(out["table"]) == 3 and out["n_compared"] == 3


def test_a_weight_left_out_in_some_rows_means_one():
    obs = Observations([{"time": 1.0, "vector": "FOPT", "value": 1.0, "sigma": 1.0},
                        {"time": 2.0, "vector": "FOPT", "value": 1.0, "sigma": 1.0, "weight": 3.0}])
    assert list(obs.table.weight) == [1.0, 3.0]


def test_weights_enter_the_objective(files, capsys):
    _, _, tmp = files
    _, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "o.json")
    by = {r["vector"]: r for r in out["table"]}
    expected = sum(r["weight"] * r["normalised"] ** 2 for r in out["table"])
    assert by["FOPT"]["weight"] == 2.0 and out["objective"] == pytest.approx(expected)


def test_full_table_is_written_with_out(files, capsys):
    _, _, tmp = files
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "o.json", "--out", tmp / "t" / "mf.json")
    rows = json.loads((tmp / "t" / "mf.json").read_text())
    assert code == 0 and out["written"] and len(rows) == 3 and {"observed", "simulated", "normalised"} <= set(rows[0])


def test_restart_misfit_equals_whole_run_misfit(files, capsys):
    _, _, tmp = files
    _cli(capsys, "run", tmp / "m.json", "--until", 10, "--state-out", tmp / "s10.json")
    save_observations(Observations([{"time": 20.0, "vector": "WWCT:P1", "value": 0.1, "sigma": 0.05}]), tmp / "late.json")
    _, whole = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "late.json")
    code, staged = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "late.json",
                        "--restart", tmp / "s10.json")
    assert code == 0 and staged["objective"] == whole["objective"]


def test_unknown_vector_is_located_with_a_hint(files, capsys):
    _, _, tmp = files
    (tmp / "bad.json").write_text(json.dumps([{"time": 5, "vector": "WBHP:P2", "value": 1.0, "sigma": 1.0}]))
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "bad.json")
    assert code == 2 and out["code"] == "unknown_vector" and "WBHP:P1" in out["allowed"]
    assert out["did_you_mean"] == {"WBHP:P2": "WBHP:P1"}


@pytest.mark.parametrize("time", [2.5, 0.0, 99.0])
def test_bad_observation_times(files, capsys, time):
    _, _, tmp = files
    (tmp / "bad.json").write_text(json.dumps([{"time": time, "vector": "FOPT", "value": 1.0, "sigma": 1.0}]))
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "bad.json")
    assert code == 2 and out["code"] == "bad_time"


def test_observation_before_the_restart_time_is_rejected(files, capsys):
    _, _, tmp = files
    _cli(capsys, "run", tmp / "m.json", "--until", 15, "--state-out", tmp / "s15.json")
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "o.json", "--restart", tmp / "s15.json")
    assert code == 2 and out["code"] == "bad_time"


def test_malformed_observations_are_located(files, capsys):
    _, _, tmp = files
    (tmp / "bad.json").write_text(json.dumps([{"time": 5, "vector": "FOPT", "value": 1.0}]))
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "bad.json")
    assert code == 2 and out["code"] == "missing_key" and out["path"] == "[0].sigma"
    code, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "nope.json")
    assert code == 2 and "not found" in out["error"]


def test_strict_flags_an_unhealthy_run(tmp_path, capsys):
    save_model(layered_waterflood(nx=3, ny=1, nz=1, days=5.0, producer_bhp=200.0), tmp_path / "m.json")
    save_observations(Observations([{"time": 5.0, "vector": "WBHP:P1", "value": 200.0, "sigma": 10.0}]), tmp_path / "o.json")
    code, out = _cli(capsys, "misfit", tmp_path / "m.json", "--observations", tmp_path / "o.json")
    assert code == 0 and out["healthy"] is False
    code, _ = _cli(capsys, "misfit", tmp_path / "m.json", "--observations", tmp_path / "o.json", "--strict")
    assert code == 1


def test_misfit_of_a_twin_is_near_zero_with_the_true_model(files, capsys):
    model, _, tmp = files
    truth = run_to(model, [10.0, 20.0]).table
    rows = [{"time": float(t), "vector": v, "value": float(truth[np.isclose(truth.TIME, t)].iloc[0][v]), "sigma": 1.0}
            for t in (10.0, 20.0) for v in ("WBHP:P1", "WOPR:P1")]
    save_observations(Observations(rows), tmp / "twin.json")
    _, out = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "twin.json")
    assert out["objective"] < 1e-12
    _cli(capsys, "edit", tmp / "m.json", "--set", "kx=20", "-o", tmp / "m2.json")
    _, worse = _cli(capsys, "misfit", tmp / "m2.json", "--observations", tmp / "twin.json")
    assert worse["objective"] > out["objective"] + 1e-6
