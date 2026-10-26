"""JSON in and out (kusim.spec) and the command line (kusim.cli)."""

import copy
import json

import numpy as np
import pytest

from conftest import LEGACY_INPUT, make_model
from kusim import (Equilibrium, Injector, Producer, ReservoirModel, Results, RockFluidTable, Simulator, __version__, load_model,
                   load_results, save_model)
from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.history import Observations
from kusim.schedule import ControlEvent, Schedule
from kusim.spec import (observations_from_list, observations_to_list, results_from_dict, schedule_from_list,
                        schedule_to_list)


def _through_json(model):
    """The model after a trip through a real JSON string (so only JSON types survive)."""
    return ReservoirModel.from_dict(json.loads(json.dumps(model.to_dict())))


def _same_run(a, b, days=5):
    ra, rb = Simulator(a).run(), Simulator(b).run()
    assert np.array_equal(ra.pressure, rb.pressure)
    assert np.array_equal(ra.sw, rb.sw)


# ---------------------------------------------------------------------- models
def test_layered_model_round_trips_exactly():
    model = layered_waterflood(nx=4, ny=3, nz=2, days=5.0)       # per-layer permeability, multi-block wells
    again = _through_json(model)
    assert np.array_equal(again.kx, model.kx) and np.array_equal(again.kz, model.kz)
    assert [w.blocks for w in again.producers] == [w.blocks for w in model.producers]
    _same_run(model, again)


def test_legacy_model_round_trips_exactly(legacy_model):
    """The thesis case: equilibrium start, capillary pressure, dipping structure, several wells."""
    again = _through_json(legacy_model)
    assert isinstance(again.initial, Equilibrium)
    assert np.array_equal(again.grid.depth, legacy_model.grid.depth)
    assert np.array_equal(again.rock_fluid.pcow, legacy_model.rock_fluid.pcow)
    short = lambda m: m.with_changes(t_end=3.0, dt=1.0)
    _same_run(short(legacy_model), short(again))


def test_uniform_values_are_written_as_one_number():
    d = make_model().to_dict()
    assert d["kx"] == 200.0 and d["porosity"] == 0.2
    assert d["initial"] == {"type": "uniform", "pressure": 3000.0, "sw": 0.2}


def test_pc_multiplier_is_folded_into_the_table():
    model = make_model(pc=True)
    rf = model.rock_fluid
    model.rock_fluid = RockFluidTable(rf.sw, rf.kro, rf.krw, rf.pcow, pc_multiplier=0.5)
    again = _through_json(model)
    assert np.allclose(again.rock_fluid.pcow, model.rock_fluid.pcow)
    assert again.rock_fluid.pc_multiplier == 1.0


def test_model_file_round_trip(tmp_path):
    model = make_model(producers=[Producer(5, "bhp", 2500.0, min_bhp=2000.0, min_oil_rate=1.0)],
                       injectors=[Injector(1, "water_rate", 100.0)])
    save_model(model, tmp_path / "m.json")
    again = load_model(tmp_path / "m.json")
    assert again.producers[0].min_bhp == 2000.0 and again.producers[0].min_oil_rate == 1.0
    assert again.provenance == model.provenance


def test_provenance_survives():
    model = make_model().with_changes(kx=150.0, note="a note")
    assert _through_json(model).provenance == ["a note"]


# ---------------------------------------------------------------------- strictness
@pytest.fixture
def base():
    return make_model(producers=[Producer(5, "bhp", 2500.0)], injectors=[Injector(1, "water_rate", 100.0)]).to_dict()


def test_unknown_model_key_is_named(base):
    base["permeability"] = 100.0
    with pytest.raises(ValueError, match="unknown key.*permeability"):
        ReservoirModel.from_dict(base)


def test_missing_model_key_is_named(base):
    del base["dt"]
    with pytest.raises(ValueError, match="missing required key.*dt"):
        ReservoirModel.from_dict(base)


def test_typo_in_a_well_option_is_an_error(base):
    base["producers"][0]["min_bph"] = 1000.0
    with pytest.raises(ValueError, match="min_bph"):
        ReservoirModel.from_dict(base)


def test_bad_initial_type_and_units_and_version(base):
    bad = copy.deepcopy(base)
    bad["initial"]["type"] = "hydrostatic"
    with pytest.raises(ValueError, match="initial.type"):
        ReservoirModel.from_dict(bad)
    bad = copy.deepcopy(base)
    bad["units"] = "metric"
    with pytest.raises(ValueError, match="oilfield"):
        ReservoirModel.from_dict(bad)
    bad = copy.deepcopy(base)
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        ReservoirModel.from_dict(bad)


def test_the_model_still_validates_the_data(base):
    base["producers"][0]["block"] = 999
    with pytest.raises(ValueError):
        ReservoirModel.from_dict(base)


# ---------------------------------------------------------------------- schedule, observations
def test_schedule_round_trip():
    schedule = Schedule([ControlEvent(0, "P1", "liquid_rate", 8000.0), ControlEvent(200, "P1", target=6000.0)])
    again = schedule_from_list(json.loads(json.dumps(schedule_to_list(schedule))))
    assert again.events == schedule.events


def test_schedule_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown key"):
        schedule_from_list([{"time": 1, "well": "P1", "rate": 5}])


def test_observations_round_trip():
    obs = Observations([{"time": 5.0, "vector": "WBHP:P1", "value": 2500.0, "sigma": 10.0}])
    again = observations_from_list(json.loads(json.dumps(observations_to_list(obs))))
    assert again.table.equals(obs.table)


# ---------------------------------------------------------------------- results
@pytest.fixture(scope="module")
def run():
    model = layered_waterflood(nx=4, ny=3, nz=2, days=5.0)
    return Results, Simulator(model).run()


def test_results_are_valid_json_without_nan(run):
    _, results = run
    assert results.wells["water_cut"].isna().any()               # injectors have no water cut: NaN in the table
    text = json.dumps(results.to_dict("all"), allow_nan=False)    # would raise on NaN
    assert "NaN" not in text


def test_results_round_trip(run, tmp_path):
    _, results = run
    results.to_json(tmp_path / "r.json", fields="all")
    again = load_results(tmp_path / "r.json")
    assert np.allclose(again.pressure, results.pressure) and np.allclose(again.times, results.times)
    assert list(again.wells.columns) == list(results.wells.columns)
    assert np.allclose(again.well("P1").bhp, results.well("P1").bhp)
    assert again.completions == results.completions


def test_field_modes(run):
    _, results = run
    n = results.grid.n_cells
    assert results.to_dict("none")["pressure"] == []
    last = results.to_dict("last")
    assert len(last["pressure"]) == 1 and len(last["pressure"][0]) == n
    assert len(results.to_dict("all")["pressure"]) == len(results.times)
    assert results_from_dict(results.to_dict("none")).pressure.shape == (0, n)
    with pytest.raises(ValueError):
        results.to_dict("some")


# ---------------------------------------------------------------------- command line
def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


def test_version(capsys):
    code, out = _cli(capsys, "version")
    assert code == 0 and out == {"ok": True, "kusim": __version__}


def test_template_validate_run(tmp_path, capsys):
    model_file = tmp_path / "m.json"
    code, out = _cli(capsys, "template", "--nx", 4, "--ny", 3, "--nz", 2, "--days", 10, "-o", model_file)
    assert code == 0 and out["n_cells"] == 24
    code, out = _cli(capsys, "validate", model_file)
    assert code == 0 and out["ok"] and out["n_steps"] == 10
    code, out = _cli(capsys, "run", model_file, "--out", tmp_path / "out", "--csv")
    assert code == 0 and out["ok"] and out["n_steps"] == 10 and out["final_time"] == 10.0
    assert set(out["wells"]) == {"P1", "I1"}
    for name in ("results.json", "summary.json", "wells.csv", "pressure.csv"):
        assert (tmp_path / "out" / name).exists()
    saved = load_results(tmp_path / "out" / "results.json")
    assert saved.pressure.shape == (1, 24)                       # the default: the last map only


def test_cli_result_equals_python_result(tmp_path, capsys):
    model = layered_waterflood(nx=4, ny=3, nz=2, days=6.0)
    save_model(model, tmp_path / "m.json")
    _cli(capsys, "run", tmp_path / "m.json", "--out", tmp_path / "o", "--fields", "all")
    again = load_results(tmp_path / "o" / "results.json")
    assert np.array_equal(again.pressure, Simulator(model).run().pressure)


def test_cli_schedule_changes_the_run(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=10.0), tmp_path / "m.json")
    (tmp_path / "s.json").write_text(json.dumps([{"time": 5, "well": "I1", "target": 50.0}]))
    _, plain = _cli(capsys, "run", tmp_path / "m.json")
    code, scheduled = _cli(capsys, "run", tmp_path / "m.json", "--schedule", tmp_path / "s.json")
    assert code == 0
    assert scheduled["wells"]["I1"]["water_rate"] == 50.0 and plain["wells"]["I1"]["water_rate"] == 150.0


def test_cli_rejects_bad_schedules(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=10.0), tmp_path / "m.json")
    (tmp_path / "s.json").write_text(json.dumps([{"time": 5, "well": "I9", "target": 50.0}]))
    code, out = _cli(capsys, "validate", tmp_path / "m.json", "--schedule", tmp_path / "s.json")
    assert code == 2 and not out["ok"] and "I9" in out["error"]
    (tmp_path / "s.json").write_text(json.dumps([{"time": 2.5, "well": "I1", "target": 50.0}]))
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--schedule", tmp_path / "s.json")
    assert code == 2 and "multiple" in out["error"]


def test_cli_errors_are_json_with_exit_code_2(tmp_path, capsys):
    code, out = _cli(capsys, "run", tmp_path / "missing.json")
    assert code == 2 and not out["ok"] and "not found" in out["error"]
    (tmp_path / "bad.json").write_text("{not json")
    code, out = _cli(capsys, "validate", tmp_path / "bad.json")
    assert code == 2 and "not valid JSON" in out["error"]
    (tmp_path / "bad.json").write_text('{"kx": 1}')
    code, out = _cli(capsys, "validate", tmp_path / "bad.json")
    assert code == 2 and "missing required" in out["error"]
    with pytest.raises(SystemExit) as stop:
        main(["run", str(tmp_path / "m.json"), "--solver", "magic"])
    assert stop.value.code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_strict_flags_trouble(tmp_path, capsys):
    """A producer held below the PVT table range is reported, and --strict turns it into exit code 1."""
    model = layered_waterflood(nx=3, ny=1, nz=1, days=5.0, producer_bhp=200.0)
    save_model(model, tmp_path / "m.json")
    code, out = _cli(capsys, "run", tmp_path / "m.json")
    assert code == 0 and out["ok"] and out["diagnostics"]["outside_pvt_range"] and out["healthy"] is False
    code, out = _cli(capsys, "run", tmp_path / "m.json", "--strict")
    assert code == 1


def test_convert_legacy_input(tmp_path, capsys):
    code, out = _cli(capsys, "convert", LEGACY_INPUT, "-o", tmp_path / "legacy.json")
    assert code == 0 and out["n_cells"] == 75
    model = load_model(tmp_path / "legacy.json")
    assert model.grid.n_cells == 75
