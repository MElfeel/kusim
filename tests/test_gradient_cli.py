"""kusim gradient: objective and adjoint gradient of the misfit from the command line, and the controls JSON."""

import json

import numpy as np
import pytest

from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.gradients import HistoryProblem
from kusim.history import Observations
from kusim.parameters import FieldParameter, ParameterSet
from kusim.schedule import ControlParameter, ControlSet
from kusim.schema import get_schema
from kusim.spec import (SpecError, controls_from_dict, controls_to_dict, save_model, save_observations)


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


OBS = [{"time": 10.0, "vector": "WBHP:I1", "value": 3300.0, "sigma": 20.0},
       {"time": 20.0, "vector": "WWCT:P1", "value": 0.1, "sigma": 0.05}]
PARAMS = {"parameters": [{"type": "field", "field": "kx", "ties": ["ky", "kz"], "lower": 1.0}]}
CONTROLS = {"controls": [{"well": "I1", "times": [0, 10], "initial": [150.0, 150.0], "upper": 1000.0}]}


@pytest.fixture
def files(tmp_path):
    model = layered_waterflood(nx=4, ny=3, nz=2, days=20.0)
    save_model(model, tmp_path / "m.json")
    save_observations(Observations(OBS), tmp_path / "o.json")
    (tmp_path / "p.json").write_text(json.dumps(PARAMS))
    (tmp_path / "c.json").write_text(json.dumps(CONTROLS))
    return model, tmp_path


def _python_problem(model):
    return HistoryProblem(model, Observations(OBS),
                          params=ParameterSet([FieldParameter("kx", ties=("ky", "kz"), lower=1.0)]),
                          controls=ControlSet([ControlParameter("I1", [0, 10], upper=1000.0)]),
                          initial_controls={"I1": [150.0, 150.0]})


# ---------------------------------------------------------------------- controls JSON
def test_controls_round_trip():
    cs = ControlSet([ControlParameter("I1", [0, 100, 200], mode="water_rate", upper=900.0),
                     ControlParameter("P1", [0, 50], lower=None, name="p1")])
    initial = {"I1": [1.0, 2.0, 3.0], "P1": [4.0, 5.0]}
    again, again_initial = controls_from_dict(json.loads(json.dumps(controls_to_dict(cs, initial))))
    assert again_initial == initial
    assert [(p.well, p.times, p.mode, p.lower, p.upper, p.name) for p in again.parameters] == \
           [(p.well, p.times, p.mode, p.lower, p.upper, p.name) for p in cs.parameters]


def test_controls_errors_are_located():
    with pytest.raises(SpecError) as error:
        controls_from_dict({"controls": [{"well": "I1", "times": [0, 5]}]})
    assert (error.value.code, error.value.path) == ("missing_key", "controls.0.initial")
    with pytest.raises(SpecError) as error:
        controls_from_dict({"controls": [{"well": "I1", "times": [0, 5], "initial": [1.0]}]})
    assert error.value.path == "controls.0.initial"
    with pytest.raises(SpecError) as error:
        controls_from_dict({"controls": [{"well": "I1", "times": [0], "initial": [1.0], "uper": 3}]})
    assert error.value.details["did_you_mean"] == {"uper": "upper"}


def test_controls_schema_accepts_the_example():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(CONTROLS, get_schema("controls"))
    jsonschema.validate(PARAMS, get_schema("parameters"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"controls": [{"well": "I1", "times": [0]}]}, get_schema("controls"))


# ---------------------------------------------------------------------- the command
def test_gradient_equals_the_python_problem(files, capsys):
    model, tmp = files
    code, out = _cli(capsys, "gradient", tmp / "m.json", "--observations", tmp / "o.json", "--parameters", tmp / "p.json",
                     "--controls", tmp / "c.json", "--out", tmp / "g.json")
    problem = _python_problem(model)
    value, grad, _ = problem.value_and_gradient(problem.vector0())
    assert code == 0 and out["ok"] and out["objective"] == pytest.approx(value, rel=1e-12)
    saved = json.loads((tmp / "g.json").read_text())
    assert np.allclose(saved["gradient"], grad, rtol=1e-10, atol=0) and saved["names"] == {"log_kx": [0, 24],
                                                                                            "I1_targets": [24, 26]}
    assert out["n_variables"] == 26 and out["variables"]["I1_targets"]["gradient"] == pytest.approx(list(grad[24:]))
    assert "gradient" not in out["variables"]["log_kx"]                    # 24 values: only in --out
    assert out["variables"]["log_kx"]["norm"] == pytest.approx(np.linalg.norm(grad[:24]))


def test_check_against_finite_differences(files, capsys):
    _, tmp = files
    code, out = _cli(capsys, "gradient", tmp / "m.json", "--observations", tmp / "o.json", "--parameters", tmp / "p.json",
                     "--check", 3)
    assert code == 0 and out["check"]["n_stable"] >= 1 and out["check"]["max_relative_error"] < 1e-4


def test_finite_difference_method_agrees_for_the_controls(files, capsys):
    _, tmp = files
    args = ("gradient", tmp / "m.json", "--observations", tmp / "o.json", "--controls", tmp / "c.json")
    _, adjoint = _cli(capsys, *args)
    _, fd = _cli(capsys, *args, "--method", "fd")
    a, f = adjoint["variables"]["I1_targets"], fd["variables"]["I1_targets"]
    assert f["gradient"][0] == pytest.approx(a["gradient"][0], rel=1e-3) and fd["diagnostics"] is None
    assert fd["objective"] == pytest.approx(adjoint["objective"], rel=1e-9)


def test_scalar_parameters_need_the_fd_method(files, capsys):
    _, tmp = files
    (tmp / "s.json").write_text(json.dumps({"parameters": [{"type": "scalar", "path": "rock_compressibility"}]}))
    args = ("gradient", tmp / "m.json", "--observations", tmp / "o.json", "--parameters", tmp / "s.json")
    code, out = _cli(capsys, *args)
    assert code == 2 and out["code"] == "no_gradient"
    code, out = _cli(capsys, *args, "--method", "fd")
    assert code == 0 and out["n_variables"] == 1 and len(out["variables"]) == 1


def test_evaluate_at_a_given_vector(files, capsys):
    model, tmp = files
    problem = _python_problem(model)
    x = problem.vector0()
    x[24:] = [120.0, 100.0]
    (tmp / "x.json").write_text(json.dumps(x.tolist()))
    _, out = _cli(capsys, "gradient", tmp / "m.json", "--observations", tmp / "o.json", "--parameters", tmp / "p.json",
                  "--controls", tmp / "c.json", "--x", tmp / "x.json")
    assert out["objective"] == pytest.approx(problem.value_and_gradient(x)[0], rel=1e-12)
    assert out["variables"]["I1_targets"]["x"] == [120.0, 100.0]
    (tmp / "x.json").write_text("[1, 2]")
    code, out = _cli(capsys, "gradient", tmp / "m.json", "--observations", tmp / "o.json", "--parameters", tmp / "p.json",
                     "--x", tmp / "x.json")
    assert code == 2 and out["code"] == "invalid_value"


def test_gradient_with_a_restart_state(files, capsys):
    _, tmp = files
    _cli(capsys, "run", tmp / "m.json", "--until", 10, "--state-out", tmp / "s10.json")
    save_observations(Observations([OBS[1]]), tmp / "late.json")
    code, out = _cli(capsys, "gradient", tmp / "m.json", "--observations", tmp / "late.json", "--parameters",
                     tmp / "p.json", "--restart", tmp / "s10.json")
    assert code == 0 and out["diagnostics"]["steps"] == 10


def test_bad_inputs_are_located(files, capsys):
    _, tmp = files
    base = ("gradient", tmp / "m.json", "--observations", tmp / "o.json")
    code, out = _cli(capsys, *base)
    assert code == 2 and out["code"] == "bad_argument"
    bad = {"controls": [{"well": "I7", "times": [0], "initial": [1.0]}]}
    (tmp / "b.json").write_text(json.dumps(bad))
    code, out = _cli(capsys, *base, "--controls", tmp / "b.json")
    assert code == 2 and out["code"] == "unknown_well" and out["path"] == "controls.0.well"
    bad["controls"][0].update(well="I1", times=[2.5])
    (tmp / "b.json").write_text(json.dumps(bad))
    code, out = _cli(capsys, *base, "--controls", tmp / "b.json")
    assert code == 2 and out["code"] == "bad_time"
    (tmp / "b.json").write_text(json.dumps({"parameters": [{"type": "field", "field": "nothing"}]}))
    code, out = _cli(capsys, *base, "--parameters", tmp / "b.json")
    assert code == 2 and out["ok"] is False
    code, out = _cli(capsys, *base, "--parameters", tmp / "p.json", "--method", "fd", "--check", 2)
    assert code == 2 and out["code"] == "bad_argument"
