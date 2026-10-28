"""Restart states and parameter sets as JSON, and staged runs from the command line."""

import json

import numpy as np
import pytest

from kusim import Simulator
from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.parameters import FieldParameter, ParameterSet, ScalarParameter
from kusim.schedule import ControlEvent, Schedule, advance
from kusim.schema import get_schema
from kusim.simulator import SimState
from kusim.spec import (SpecError, load_results, parameters_from_dict, parameters_to_dict, save_model, save_schedule,
                        state_from_dict, state_to_dict)


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def model():
    return layered_waterflood(nx=4, ny=3, nz=2, days=12.0)


def _through_json(state):
    return state_from_dict(json.loads(json.dumps(state_to_dict(state), allow_nan=False)))


# ---------------------------------------------------------------------- SimState
def test_state_round_trips_exactly(model):
    sim = Simulator(model)
    advance(sim, 6.0)
    state = sim.get_state()
    again = _through_json(state)
    assert again.time == state.time
    assert np.array_equal(again.p_o, state.p_o) and np.array_equal(again.s_w, state.s_w)
    assert again.well_controls == state.well_controls and again.cum == state.cum


def test_run_continued_from_a_json_state_equals_one_run(model):
    whole = Simulator(model)
    whole.run()
    first = Simulator(model)
    advance(first, 6.0)
    second = Simulator(model)
    second.set_state(_through_json(first.get_state()))
    advance(second, 12.0)
    assert np.array_equal(second.p_o, whole.p_o) and np.array_equal(second.s_w, whole.s_w)


def test_state_keeps_a_changed_well_control(model):
    sim = Simulator(model)
    sim.set_well_control("P1", "bhp", 2100.0)
    again = _through_json(sim.get_state())
    assert again.well_controls["P1"] == ("bhp", 2100.0)


def test_state_errors_are_located(model):
    data = state_to_dict(Simulator(model).get_state())
    with pytest.raises(SpecError) as error:
        state_from_dict(dict(data, presure=[]))
    assert error.value.code == "unknown_key"
    del data["cum"]
    with pytest.raises(SpecError) as error:
        state_from_dict(data)
    assert (error.value.code, error.value.path) == ("missing_key", "cum")


def test_sim_state_methods(model, tmp_path):
    state = Simulator(model).get_state()
    state.to_json(tmp_path / "s.json")
    assert np.array_equal(SimState.from_json(tmp_path / "s.json").p_o, state.p_o)
    assert np.array_equal(SimState.from_dict(state.to_dict()).s_w, state.s_w)


# ---------------------------------------------------------------------- ParameterSet
def test_parameter_set_round_trips():
    params = ParameterSet([FieldParameter("kx", transform="log", lower=1.0, ties=("ky", "kz")),
                           FieldParameter("porosity", transform="logit", lower=0.02, upper=0.4, name="phi"),
                           ScalarParameter("initial.woc_depth", lower=0.0)])
    again = parameters_from_dict(json.loads(json.dumps(parameters_to_dict(params))))
    assert parameters_to_dict(again) == parameters_to_dict(params)
    assert list(again.parameters[0].ties) == ["ky", "kz"]


def test_parameter_set_gives_the_same_vector(model):
    params = ParameterSet([FieldParameter("kx", ties=("ky", "kz"), lower=1.0), ScalarParameter("rock_compressibility")])
    again = ParameterSet.from_dict(params.to_dict())
    assert np.array_equal(again.vector(model), params.vector(model))


def test_parameter_errors_are_located():
    with pytest.raises(SpecError) as error:
        parameters_from_dict({"parameters": [{"type": "field"}]})
    assert (error.value.code, error.value.path) == ("missing_key", "parameters.0.field")
    with pytest.raises(SpecError) as error:
        parameters_from_dict({"parameters": [{"type": "field", "field": "kx", "transfrom": "log"}]})
    assert error.value.details["did_you_mean"] == {"transfrom": "transform"}
    with pytest.raises(SpecError):
        parameters_from_dict({"parameters": [{"type": "cube"}]})
    with pytest.raises(ValueError):                              # the class itself still checks the values
        parameters_from_dict({"parameters": [{"type": "field", "field": "kx", "transform": "logit"}]})


# ---------------------------------------------------------------------- schemas
def test_state_and_parameter_schemas_validate_examples(model):
    jsonschema = pytest.importorskip("jsonschema")
    state = json.loads(json.dumps(state_to_dict(Simulator(model).get_state())))
    jsonschema.validate(state, get_schema("state"))
    params = ParameterSet([FieldParameter("kx", ties=("ky",), lower=1.0), ScalarParameter("initial.sw")])
    jsonschema.validate(json.loads(json.dumps(parameters_to_dict(params))), get_schema("parameters"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(state, extra=1), get_schema("state"))


# ---------------------------------------------------------------------- command line
def test_staged_run_equals_single_run(model, tmp_path, capsys):
    save_model(model, tmp_path / "m.json")
    _cli(capsys, "run", tmp_path / "m.json", "--out", tmp_path / "whole", "--state-out", tmp_path / "whole.json")
    code, first = _cli(capsys, "run", tmp_path / "m.json", "--until", 5, "--state-out", tmp_path / "s5.json")
    assert code == 0 and first["final_time"] == 5.0 and first["start_time"] == 0.0 and first["n_steps"] == 5
    code, second = _cli(capsys, "run", tmp_path / "m.json", "--restart", tmp_path / "s5.json",
                        "--state-out", tmp_path / "s12.json")
    assert code == 0 and second["start_time"] == 5.0 and second["final_time"] == 12.0 and second["n_steps"] == 12
    whole, staged = (json.loads((tmp_path / name).read_text()) for name in ("whole.json", "s12.json"))
    assert whole["pressure"] == staged["pressure"] and whole["sw"] == staged["sw"] and whole["cum"] == staged["cum"]


def test_staged_run_with_one_schedule_for_all_stages(model, tmp_path, capsys):
    save_model(model, tmp_path / "m.json")
    save_schedule(Schedule([ControlEvent(3.0, "I1", target=60.0), ControlEvent(8.0, "I1", target=30.0)]),
                  tmp_path / "s.json")
    _cli(capsys, "run", tmp_path / "m.json", "--schedule", tmp_path / "s.json", "--state-out", tmp_path / "whole.json")
    _cli(capsys, "run", tmp_path / "m.json", "--schedule", tmp_path / "s.json", "--until", 5,
         "--state-out", tmp_path / "a.json")
    _cli(capsys, "run", tmp_path / "m.json", "--schedule", tmp_path / "s.json", "--restart", tmp_path / "a.json",
         "--until", 9, "--state-out", tmp_path / "b.json")
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--schedule", tmp_path / "s.json",
                     "--restart", tmp_path / "b.json", "--state-out", tmp_path / "c.json")
    assert code == 0 and out["wells"]["I1"]["water_rate"] == 30.0
    assert json.loads((tmp_path / "whole.json").read_text()) == json.loads((tmp_path / "c.json").read_text())


def test_restart_changes_a_well_in_the_second_stage(model, tmp_path, capsys):
    save_model(model, tmp_path / "m.json")
    _cli(capsys, "run", tmp_path / "m.json", "--until", 5, "--state-out", tmp_path / "s5.json")
    save_schedule(Schedule([ControlEvent(5.0, "I1", target=40.0)]), tmp_path / "s.json")
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--restart", tmp_path / "s5.json",
                     "--schedule", tmp_path / "s.json", "--out", tmp_path / "out")
    assert code == 0 and out["wells"]["I1"]["water_rate"] == 40.0
    assert load_results(tmp_path / "out" / "results.json").wells["time"].min() > 5.0


def test_validate_checks_a_restart_state(model, tmp_path, capsys):
    save_model(model, tmp_path / "m.json")
    _cli(capsys, "run", tmp_path / "m.json", "--until", 5, "--state-out", tmp_path / "s5.json")
    code, out = _cli(capsys, "validate", tmp_path / "m.json", "--restart", tmp_path / "s5.json")
    assert code == 0 and out["restart_time"] == 5.0
    other = layered_waterflood(nx=3, ny=3, nz=2, days=12.0)
    save_model(other, tmp_path / "other.json")
    code, out = _cli(capsys, "validate", tmp_path / "other.json", "--restart", tmp_path / "s5.json")
    assert code == 2 and out["code"] == "state_mismatch"


def test_bad_until_and_missing_state_are_errors(model, tmp_path, capsys):
    save_model(model, tmp_path / "m.json")
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--until", 99)
    assert code == 2 and out["code"] == "bad_argument"
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--until", 2.5)
    assert code == 2 and "whole number of steps" in out["error"]
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--restart", tmp_path / "nope.json")
    assert code == 2 and "not found" in out["error"]
