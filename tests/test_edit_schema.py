"""Path edits, model diffs, JSON Schema and the located error messages (kusim.edit, kusim.schema, SpecError)."""

import json

import pytest

from conftest import make_model
from kusim import Injector, Producer, ReservoirModel
from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.edit import apply_edits, diff_models, get_path, parse_value, set_path
from kusim.schema import SCHEMAS, get_schema
from kusim.spec import (_EVENT_KEYS, _GRID_KEYS, _INJECTOR_KEYS, _MODEL_KEYS, _MODEL_REQUIRED, _PRODUCER_KEYS, _PVT_KEYS,
                        _ROCK_FLUID_KEYS, SpecError, model_to_dict, observations_from_list, save_model, schedule_from_list)


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def data():
    return model_to_dict(layered_waterflood(nx=4, ny=3, nz=2, days=10.0))


# ---------------------------------------------------------------------- paths and edits
def test_parse_value():
    assert parse_value("150") == 150 and parse_value("null") is None and parse_value("[1, 2]") == [1, 2]
    assert parse_value("bhp") == "bhp" and parse_value('"bhp"') == "bhp"


def test_get_and_set_by_path(data):
    assert get_path(data, "producers.0.mode") == data["producers"][0]["mode"]
    old = set_path(data, "producers.0.target", 2600.0)
    assert old == 2500.0 and data["producers"][0]["target"] == 2600.0
    assert set_path(data, "pvt.bo.1", 1.11) == model_to_dict(layered_waterflood(nx=4, ny=3, nz=2, days=10.0))["pvt"]["bo"][1]


@pytest.mark.parametrize("path", ["producers.9.target", "dt.x", "kx..3", "", "producers.x"])
def test_bad_paths_are_located(data, path):
    with pytest.raises(SpecError) as error:
        set_path(data, path, 1)
    assert error.value.code == "bad_path"


def test_edit_gives_the_edited_model_and_the_changes(data):
    model, normalised, changes = apply_edits(data, [("kx", 150.0), ("producers.0.target", 2600.0)])
    assert (model.kx == 150.0).all() and model.producers[0].target == 2600.0
    assert normalised["kx"] == 150.0
    assert [c["path"] for c in changes] == ["kx", "producers.0.target"]
    assert changes[0]["old"]["n"] == len(data["kx"])            # a long list is summarised
    assert data["producers"][0]["target"] == 2500.0               # the input is not modified


def test_edit_rejects_a_typo_and_names_the_fix(data):
    with pytest.raises(SpecError) as error:
        apply_edits(data, [("producers.0.min_bph", 1000.0)])
    assert error.value.code == "unknown_key" and error.value.path == "producers.0.min_bph"
    assert error.value.details["did_you_mean"] == {"min_bph": "min_bhp"}


def test_a_new_top_level_key_is_rejected_when_the_model_is_rebuilt(data):
    with pytest.raises(SpecError) as error:
        apply_edits(data, [("porosty", 0.3)])
    assert error.value.code == "unknown_key" and error.value.details["did_you_mean"] == {"porosty": "porosity"}


def test_edit_rejects_an_invalid_model(data):
    with pytest.raises(ValueError):
        apply_edits(data, [("producers.0.block", 9999)])
    with pytest.raises(ValueError):
        apply_edits(data, [("producers.0.mode", "magic")])


def test_merge_patch_merges_dicts_and_replaces_lists(data):
    patch = {"initial": {"sw": 0.25}, "injectors": [dict(data["injectors"][0], target=99.0)]}
    model, normalised, changes = apply_edits(data, [], patch)
    assert model.initial.sw == 0.25 and model.initial.pressure == data["initial"]["pressure"]
    assert model.injectors[0].target == 99.0
    assert {c["path"] for c in changes} == {"initial.sw", "injectors"}


def test_setting_one_block_of_a_uniform_array_explains_itself(data):
    uniform = dict(data, kx=100.0)
    with pytest.raises(SpecError, match="single value"):
        apply_edits(uniform, [("kx.3", 500.0)])
    model, _, _ = apply_edits(data, [("kx.3", 500.0)])           # a per-block list can be edited by position
    assert model.kx[3] == 500.0


# ---------------------------------------------------------------------- diff
def test_diff_of_equal_models_is_empty():
    model = make_model()
    assert diff_models(model, model.with_changes()) == []


def test_diff_lists_changes_by_path():
    a = make_model(producers=[Producer(5, "bhp", 2500.0)], injectors=[Injector(1, "water_rate", 100.0)])
    b = a.with_changes(dt=0.5)
    b.producers[0] = Producer(5, "bhp", 2400.0)
    changes = {c["path"]: (c["a"], c["b"]) for c in diff_models(a, b)}
    assert changes == {"dt": (a.dt, 0.5), "producers.0.target": (2500.0, 2400.0)}


def test_diff_summarises_a_long_array_change():
    a = layered_waterflood(nx=4, ny=3, nz=2, days=5.0)
    b = a.with_changes(kx=a.kx * 2.0)
    (change,) = [c for c in diff_models(a, b) if c["path"] == "kx"]
    assert change["n_changed"] == 24 and change["of"] == 24


# ---------------------------------------------------------------------- the located errors
def test_errors_carry_path_and_code(data):
    data["pvt"]["bx"] = [1.0]
    with pytest.raises(SpecError) as error:
        ReservoirModel.from_dict(data)
    assert (error.value.code, error.value.path) == ("unknown_key", "pvt.bx")
    del data["pvt"]["bx"], data["dt"]
    with pytest.raises(SpecError) as error:
        ReservoirModel.from_dict(data)
    assert (error.value.code, error.value.path) == ("missing_key", "dt")


def test_schedule_and_observation_errors_name_the_item():
    with pytest.raises(SpecError) as error:
        schedule_from_list([{"time": 1, "well": "P1", "target": 1}, {"time": 1, "well": "P1", "rate": 5}])
    assert error.value.path == "[1].rate"
    with pytest.raises(SpecError) as error:
        observations_from_list([{"time": 1, "vector": "WBHP:P1", "value": 1.0}])
    assert (error.value.code, error.value.path) == ("missing_key", "[0].sigma")


def test_cli_edit_diff_and_schema(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=10.0), tmp_path / "m.json")
    code, out = _cli(capsys, "edit", tmp_path / "m.json", "--set", "producers.0.target=2600", "--set", "dt=0.5",
                     "-o", tmp_path / "n.json")
    assert code == 0 and out["ok"] and out["dt"] == 0.5
    assert {c["path"] for c in out["changes"]} == {"producers.0.target", "dt"}
    code, out = _cli(capsys, "diff", tmp_path / "m.json", tmp_path / "n.json")
    assert code == 0 and out["n_differences"] == 2 and not out["identical"]
    code, out = _cli(capsys, "diff", tmp_path / "m.json", tmp_path / "m.json")
    assert out["identical"]
    code, out = _cli(capsys, "edit", tmp_path / "m.json", "--set", "dt=0.5", "--in-place")
    assert code == 0 and json.loads((tmp_path / "m.json").read_text())["dt"] == 0.5
    code, out = _cli(capsys, "schema", "schedule")
    assert code == 0 and out["title"] == "kusim schedule"


def test_cli_edit_errors_are_structured(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=10.0), tmp_path / "m.json")
    code, out = _cli(capsys, "edit", tmp_path / "m.json", "--set", "producers.0.min_bph=1", "-o", tmp_path / "n.json")
    assert code == 2 and out["code"] == "unknown_key" and out["did_you_mean"] == {"min_bph": "min_bhp"}
    code, out = _cli(capsys, "edit", tmp_path / "m.json", "--set", "producers.3.target=1", "-o", tmp_path / "n.json")
    assert code == 2 and out["code"] == "bad_path" and out["path"] == "producers.3.target"
    code, out = _cli(capsys, "edit", tmp_path / "m.json", "--set", "nonsense", "-o", tmp_path / "n.json")
    assert code == 2 and out["code"] == "bad_argument"
    assert not (tmp_path / "n.json").exists()                     # a rejected edit writes nothing
    (tmp_path / "bad.json").write_text('{"kx": 1,')
    code, out = _cli(capsys, "validate", tmp_path / "bad.json")
    assert code == 2 and out["code"] == "invalid_json" and "line" in out
    (tmp_path / "bad.json").write_text("{}")
    code, out = _cli(capsys, "validate", tmp_path / "bad.json")
    assert out["code"] == "missing_key" and out["path"] in _MODEL_REQUIRED


# ---------------------------------------------------------------------- schema
def _keys(schema):
    return set(schema["properties"]), set(schema["required"])


def test_schema_keys_match_the_readers():
    model = get_schema("model")
    props = model["properties"]
    assert _keys(model) == (_MODEL_KEYS, _MODEL_REQUIRED)
    assert set(props["grid"]["properties"]) == _GRID_KEYS
    assert set(props["pvt"]["properties"]) == _PVT_KEYS
    assert set(props["rock_fluid"]["properties"]) == _ROCK_FLUID_KEYS
    assert set(props["producers"]["items"]["properties"]) == _PRODUCER_KEYS
    assert set(props["injectors"]["items"]["properties"]) == _INJECTOR_KEYS
    assert set(get_schema("schedule")["items"]["properties"]) == _EVENT_KEYS
    assert _keys(get_schema("observations")["items"])[0] == {"time", "vector", "value", "sigma", "weight"}


def test_every_schema_is_json_and_a_valid_schema():
    jsonschema = pytest.importorskip("jsonschema")
    for name in SCHEMAS:
        schema = json.loads(json.dumps(get_schema(name)))
        jsonschema.Draft202012Validator.check_schema(schema)


def test_example_files_validate_against_the_schemas(data):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json.loads(json.dumps(data)), get_schema("model"))
    jsonschema.validate(json.loads(json.dumps(model_to_dict(make_model(pc=True)))), get_schema("model"))
    jsonschema.validate([{"time": 5, "well": "I1", "target": 50.0}], get_schema("schedule"))
    jsonschema.validate([{"time": 5, "vector": "WBHP:P1", "value": 2500.0, "sigma": 10.0}], get_schema("observations"))
    broken = dict(data, kx="high")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, get_schema("model"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(data, extra=1), get_schema("model"))


def test_results_validate_against_the_results_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from kusim import Simulator

    results = Simulator(layered_waterflood(nx=4, ny=3, nz=2, days=3.0)).run()
    jsonschema.validate(json.loads(json.dumps(results.to_dict("last"), allow_nan=False)), get_schema("results"))
