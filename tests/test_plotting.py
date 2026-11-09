"""kusim tables / plot: the query language over kusim files, the Plotly binding and its refusals, and the worked examples of docs/plotting.md."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kusim import Producer, save_model
from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.figure import check_figure, prepare
from kusim.queries import Matrix, run_tables, source_from_data
from kusim.spec import SpecError, _read_json

EXAMPLES = Path(__file__).resolve().parent.parent / "src" / "kusim" / "plot_examples"


def cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """model.json (two producers), run/results.json (all snapshots), observations.json (noisy water cut and pressure of both
    producers from a run of a different model) and misfit.json (that model's mismatch)."""
    import os

    folder = tmp_path_factory.mktemp("plots")
    old = Path.cwd()
    os.chdir(folder)
    try:
        model = layered_waterflood(nx=6, ny=4, nz=3, days=60.0)
        p1 = model.producers[0]
        model = dataclasses.replace(model, producers=[p1, Producer(block=model.grid.column_blocks(1, 4), mode="bhp", target=p1.target)])
        save_model(model, "model.json")
        assert main(["run", "model.json", "--out", "run", "--fields", "all"]) == 0
        results = _read_json("run/results.json")
        rng = np.random.default_rng(3)
        truth = model.with_changes(kx=model.kx * 0.5, ky=model.ky * 0.5)
        save_model(truth, "truth.json")
        assert main(["run", "truth.json", "--out", "truth"]) == 0
        series = source_from_data(_read_json("truth/results.json")).table("series")
        rows = []
        for r in series[series.time.isin(np.arange(10.0, 61.0, 10.0)) & series.quantity.isin(["WWCT", "WBHP"])].itertuples():
            sigma = 0.02 if r.quantity == "WWCT" else 10.0
            rows.append({"time": float(r.time), "vector": r.vector, "value": float(r.value + sigma * rng.standard_normal()), "sigma": sigma})
        Path("observations.json").write_text(json.dumps(rows))
        assert main(["misfit", "model.json", "--observations", "observations.json", "--out", "misfit.json"]) == 0
        assert results["snapshot_times"][-1] == 60.0
        yield folder
    finally:
        os.chdir(old)


@pytest.fixture
def ws(workspace, monkeypatch):
    monkeypatch.chdir(workspace)
    return workspace


# ---------------------------------------------------------------------- the tables of each kind of file
def test_tables_of_every_kind_of_file(ws, capsys):
    code, model = cli(capsys, "tables", "model.json")
    assert code == 0 and model["kind"] == "model" and model["tables"]["cells"]["n_rows"] == 72
    wells = model["tables"]["wells"]
    assert wells["n_rows"] == 3 and {"well", "x", "y", "mode", "target"} <= set(wells["columns"])
    code, run = cli(capsys, "tables", "run/results.json")
    assert run["kind"] == "results" and run["tables"]["snapshots"]["saved_times"]["last"] == 60.0
    assert run["tables"]["snapshots"]["n_rows"] == 72 * 61 and "WWCT" in run["tables"]["series"]["quantities"]
    assert run["tables"]["series"]["quantity_units"]["WWCT"] == "fraction"
    assert cli(capsys, "tables", "observations.json")[1]["kind"] == "observations"
    assert cli(capsys, "tables", "misfit.json")[1]["kind"] == "residuals"
    assert cli(capsys, "tables", "run/summary.json")[1]["tables"]["summary_wells"]["n_rows"] == 3
    code, bad = cli(capsys, "tables", "model.json.nope")
    assert code == 2 and bad["code"] == "file_not_found"
    Path("other.json").write_text('{"a": 1}')
    code, bad = cli(capsys, "tables", "other.json")
    assert code == 2 and bad["code"] == "bad_source"


# ---------------------------------------------------------------------- queries
def sources(ws):
    return {"model": source_from_data(_read_json("model.json")), "run": source_from_data(_read_json("run/results.json")),
            "obs": source_from_data(_read_json("observations.json")), "fit": source_from_data(_read_json("misfit.json"))}


def test_where_group_weighted_mean_and_pivot(ws):
    src = sources(ws)
    t = run_tables(src, {
        "last": {"from": "run.snapshots", "where": [{"column": "time", "op": "==", "value": "last"}]},
        "avg": {"from": "last", "join": {"table": "model.cells", "on": "block"}, "group_by": ["k"],
                "aggregate": {"sw": {"fn": "mean", "column": "sw", "weight": "pore_volume"}, "n": {"fn": "count"}}},
        "grid": {"from": "last", "group_by": ["x", "y"], "aggregate": {"sw": {"fn": "mean", "column": "sw"}}, "pivot": {"rows": "y", "columns": "x", "values": "sw"}}}).derived
    assert len(t["last"]) == 72 and list(t["avg"].k) == [1, 2, 3] and set(t["avg"].n) == {24}
    cells, last = src["model"].table("cells"), t["last"].set_index("block")
    layer = cells[cells.k == 2]
    expected = (last.loc[layer.block, "sw"].to_numpy() * layer.pore_volume.to_numpy()).sum() / layer.pore_volume.sum()
    assert t["avg"].set_index("k").loc[2, "sw"] == pytest.approx(expected)
    assert isinstance(t["grid"], Matrix) and t["grid"].shape == [4, 6] and len(t["grid"].x) == 6


def test_only_the_selected_snapshots_are_built(ws):
    src = sources(ws)
    t = run_tables(src, {"one": {"from": "run.snapshots", "where": [{"column": "time", "op": "in", "value": [0, 30, "last"]}]}}).derived["one"]
    assert sorted(t.time.unique()) == [0.0, 30.0, 60.0] and len(t) == 3 * 72
    with pytest.raises(SpecError) as error:                                         # all snapshots at once would be too many rows
        import kusim.queries as q
        old, q.MAX_ROWS = q.MAX_ROWS, 1000
        try:
            run_tables(src, {"all": {"from": "run.snapshots"}})
        finally:
            q.MAX_ROWS = old
    assert error.value.code == "too_large"


def test_join_of_observed_and_simulated_and_rms_per_well(ws):
    src = sources(ws)
    t = run_tables(src, {
        "wwct": {"from": "obs.observations", "where": {"column": "quantity", "op": "==", "value": "WWCT"}},
        "both": {"from": "wwct", "join": {"table": "run.series", "on": ["time", "vector"], "suffix": "_sim"}},
        "rms": {"from": "fit.residuals", "group_by": ["well", "quantity"], "aggregate": {"rms": {"fn": "rms", "column": "normalised"}}, "sort_by": "-rms"}}).derived
    assert len(t["both"]) == len(t["wwct"]) == 12 and {"value", "value_sim"} <= set(t["both"].columns)
    assert t["rms"].rms.is_monotonic_decreasing and set(t["rms"].well) == {"I1", "P1", "P2"}
    direct = src["fit"].table("residuals")
    one = direct[(direct.well == "P1") & (direct.quantity == "WWCT")].normalised
    row = t["rms"][(t["rms"].well == "P1") & (t["rms"].quantity == "WWCT")].rms.iloc[0]
    assert row == pytest.approx(float(np.sqrt((one ** 2).mean())))


@pytest.mark.parametrize("query, code, fragment", [
    ({"from": "run.nothing"}, "unknown_table", "series"),
    ({"from": "nowhere.series"}, "unknown_table", "available"),
    ({"from": "run.series", "where": [{"column": "wel", "op": "==", "value": "P1"}]}, "unknown_column", "well"),
    ({"from": "run.series", "where": [{"column": "well", "op": "~", "value": "P1"}]}, "bad_value", "=="),
    ({"from": "run.series", "group_by": ["well"]}, "missing_key", "aggregate"),
    ({"from": "run.series", "aggregate": {"x": {"fn": "average", "column": "value"}}}, "bad_value", "mean"),
    ({"from": "run.series", "aggregate": {"x": {"fn": "sum", "column": "valu"}}}, "unknown_column", "value"),
    ({"from": "run.series", "pivot": {"rows": "time", "columns": "well", "values": "value"}}, "pivot_duplicates", "Aggregate first"),
    ({"from": "run.series", "sorted": "time"}, "unknown_key", "sort_by"),
    ({"from": "run.series", "where": [{"column": "well", "op": "==", "value": "P1"}], "limit": -1}, "bad_value", "whole number"),
])
def test_query_errors_say_what_to_fix(ws, query, code, fragment):
    with pytest.raises(SpecError) as error:
        run_tables(sources(ws), {"t": query})
    assert error.value.code == code and fragment in str(error.value) + json.dumps(error.value.details)


# ---------------------------------------------------------------------- figures
def figure_spec(figure, tables=None, **extra):
    return {"sources": {"run": "run/results.json", "model": "model.json"}, **extra, "tables": tables or {}, "figure": figure}


def test_each_makes_one_trace_per_group_and_filters_every_table_with_the_column(ws):
    spec = json.loads((EXAMPLES / "observed_vs_simulated.json").read_text())
    bound, _, info = prepare(spec)
    names = [t["name"] for t in bound["data"]]
    assert names == ["simulated P1", "simulated P2", "observed P1", "observed P2"] and info["figure"]["n_traces"] == 4
    sim_p1, obs_p1 = bound["data"][0], bound["data"][2]
    assert len(sim_p1["x"]) == 60 and len(obs_p1["x"]) == 6 and len(obs_p1["error_y"]["array"]) == 6


def test_refs_to_unknown_things_are_located(ws):
    for ref, code in (("nothing.x", "unknown_ref"), ("wc.valu", "unknown_column"), ("wc", "bad_ref"), ("wc.value.extra", "bad_ref")):
        spec = figure_spec({"data": [{"type": "scatter", "y": {"$ref": ref}}]}, {"wc": {"from": "run.series"}})
        with pytest.raises(SpecError) as error:
            prepare(spec)
        assert error.value.code == code and error.value.path.startswith("figure.data.0"), (ref, error.value)
    spec = figure_spec({"data": [{"type": "scatter", "y": {"$ref": "g.value"}}]}, {"g": {"from": "run.series", "pivot": {"rows": "time", "columns": "vector", "values": "value"}}})
    with pytest.raises(SpecError) as error:
        prepare(spec)
    assert "pivoted" in str(error.value)
    with pytest.raises(SpecError):
        prepare(figure_spec({"data": [{"type": "scatter", "name": "$group"}]}))


def test_a_source_table_can_be_read_directly_and_an_empty_trace_is_reported(ws):
    bound, _, info = prepare(figure_spec({"data": [{"type": "scatter", "x": {"$ref": "run.series.time"}, "y": {"$ref": "run.series.value"}}]}))
    assert len(bound["data"][0]["x"]) == 780      # 60 steps x (4 + 4 + 2 well vectors + 3 field vectors)
    spec = figure_spec({"data": [{"type": "scatter", "x": {"$ref": "none.time"}}]}, {"none": {"from": "run.series", "where": [{"column": "well", "op": "==", "value": "P9"}]}})
    assert "has no rows" in prepare(spec)[2]["warnings"][0]


@pytest.mark.parametrize("figure, fragment", [
    ({"data": [{"type": "scattermapbox", "lat": [1], "lon": [1]}]}, "not accepted"),
    ({"data": [{"type": "scatter"}], "layout": {"mapbox": {"style": "open-street-map"}}}, "layout.mapbox"),
    ({"data": [{"type": "scatter"}], "layout": {"title": {"text": "see https://example.com/x"}}}, "link"),
    ({"data": [{"type": "scatter"}], "layout": {"images": [{"source": "http://example.com/a.png"}]}}, "link"),
    ({"data": [{"type": "scatter", "text": ["<a href='javascript:alert(1)'>x</a>"]}]}, "link"),
])
def test_figures_may_not_fetch_anything(ws, figure, fragment):
    with pytest.raises(SpecError) as error:
        prepare(figure_spec(figure))
    assert error.value.code == "forbidden_content" and fragment in str(error.value)


def test_plotly_validation_names_the_bad_property_and_too_much_data_is_refused(ws, capsys):
    Path("bad.json").write_text(json.dumps(figure_spec({"data": [{"type": "scatter", "mode": "lines", "colour": "red"}]})))
    code, answer = cli(capsys, "plot", "bad.json", "-o", "x.json")
    assert code == 2 and answer["code"] == "invalid_figure" and "colour" in answer["error"] and not Path("x.json").exists()
    import kusim.figure as fg
    old, fg.MAX_NUMBERS = fg.MAX_NUMBERS, 100
    try:
        with pytest.raises(SpecError) as error:
            prepare(figure_spec({"data": [{"type": "scatter", "y": {"$ref": "run.series.value"}}]}))
    finally:
        fg.MAX_NUMBERS = old
    assert error.value.code == "too_large"


def test_files_named_in_a_description_stay_in_the_working_folder(ws):
    for name in ("../model.json", "C:/model.json", str(Path("model.json").resolve()), "a/../../model.json"):
        with pytest.raises(SpecError) as error:
            prepare({"sources": {"m": name}, "figure": {"data": []}})
        assert error.value.code == "bad_path"
    with pytest.raises(SpecError) as error:
        prepare({"sources": {"m": "nothing.json"}, "figure": {"data": []}})
    assert error.value.code == "file_not_found" and "files_nearby" in error.value.details


def test_unknown_keys_of_the_description_are_errors(ws):
    with pytest.raises(SpecError) as error:
        prepare({"source": {}, "figure": {"data": []}})
    assert error.value.code == "unknown_key" and error.value.details["did_you_mean"] == {"source": "sources"}


# ---------------------------------------------------------------------- the worked examples
@pytest.mark.parametrize("name", sorted(p.stem for p in EXAMPLES.glob("*.json")))
def test_every_worked_example_runs(ws, capsys, name):
    code, answer = cli(capsys, "plot", EXAMPLES / f"{name}.json", "-o", f"fig_{name}.json")
    assert code == 0, answer
    assert not answer.get("warnings") and answer["figure"]["n_traces"] >= 1
    figure = json.loads(Path(f"fig_{name}.json").read_text())
    for trace in figure["data"]:                                                   # nothing was left unbound
        assert "$ref" not in json.dumps(trace) and "$each" not in trace
    if name == "average_map":
        heat = figure["data"][0]
        assert np.array(heat["z"], dtype=float).shape == (4, 6) and len(figure["data"][2]["text"]) == 3
    if name == "mismatch_map":
        assert sorted(figure["data"][1]["text"]) == ["P1", "P2"] and all(v > 0 for v in figure["data"][1]["marker"]["size"])
    if name in ("average_map", "layer_map", "mismatch_map"):                       # the grid boundary: a closed rectangle
        outline = next(t for t in figure["data"] if t.get("name") == "model boundary")
        assert outline["x"][0] == outline["x"][-1] and outline["y"][0] == outline["y"][-1] and len(outline["x"]) == 5
    if name == "vertical_profile":
        assert figure["data"][0]["y"] == sorted(figure["data"][0]["y"]) and len(figure["data"][0]["x"]) == 3


@pytest.mark.slow
def test_an_image_and_a_page_are_written(ws, capsys):
    code, png = cli(capsys, "plot", EXAMPLES / "average_map.json", "-o", "map.png", "--width", 600, "--height", 400, "--scale", 1)
    assert code == 0 and png["format"] == "png" and Path("map.png").read_bytes()[:4] == b"\x89PNG"
    code, html = cli(capsys, "plot", EXAMPLES / "water_cut_by_well.json", "-o", "wc.html")
    assert code == 0 and "plotly" in Path("wc.html").read_text(encoding="utf8")[:200000].lower()


def test_output_must_be_a_known_format_and_a_sane_size(ws, capsys):
    code, answer = cli(capsys, "plot", EXAMPLES / "layer_map.json", "-o", "x.exe")
    assert code == 2 and "must end in" in answer["error"]
    code, answer = cli(capsys, "plot", EXAMPLES / "layer_map.json", "-o", "x.png", "--width", 100000)
    assert code == 2 and answer["code"] == "too_large"


def test_plot_template_lists_and_writes_the_examples(ws, capsys):
    from kusim.figure import EXAMPLE_INFO

    assert set(EXAMPLE_INFO) == {p.stem for p in EXAMPLES.glob("*.json")}
    code, listing = cli(capsys, "plot-template")
    assert code == 0 and {e["name"] for e in listing["examples"]} == set(EXAMPLE_INFO) and all(e["needs"] for e in listing["examples"])
    code, written = cli(capsys, "plot-template", "--name", "mismatch_map", "-o", "work/spec.json")
    assert code == 0 and Path("work/spec.json").exists() and written["needs"]
    assert cli(capsys, "plot", "work/spec.json", "-o", "work/fig.json")[0] == 0              # what was written is runnable as it is
    code, bad = cli(capsys, "plot-template", "--name", "mismatch_maps", "-o", "x.json")
    assert code == 2 and bad["code"] == "unknown_example" and bad["did_you_mean"] == {"mismatch_maps": "mismatch_map"}
    assert cli(capsys, "plot-template", "--name", "layer_map")[1]["code"] == "bad_argument"


def test_template_with_sources_quantity_well(ws, capsys):
    code, answer = cli(capsys, "plot-template", "--name", "observed_vs_simulated", "-o", "t.json", "--source", "run=truth/results.json",
                       "--quantity", "WBHP", "--well", "P2", "--title", "pressure")
    assert code == 0 and answer["sources"]["run"] == "truth/results.json"
    spec = _read_json("t.json")
    assert spec["tables"]["observed"]["where"][0]["value"] == "WBHP" and spec["tables"]["observed"]["where"][-1]["value"] == "P2"
    assert spec["figure"]["layout"]["title"] == "pressure"
    code, answer = cli(capsys, "plot", "t.json", "-o", "t.html")
    assert code == 0, answer
    code, answer = cli(capsys, "plot-template", "--name", "observed_vs_simulated", "-o", "t.json", "--source", "fit=x.json")
    assert code != 0 and answer["code"] == "unknown_source"


def test_plot_description_is_not_a_model_and_a_model_is_not_a_run(ws, capsys):
    cli(capsys, "plot-template", "--name", "observed_vs_simulated", "-o", "t.json", "--source", "run=model.json")
    code, answer = cli(capsys, "edit", "t.json", "--set", "figure.layout.title=x", "-o", "u.json")
    assert code != 0 and answer["code"] == "not_a_model"
    code, answer = cli(capsys, "plot", "t.json", "-o", "t.html")
    assert code != 0 and answer["code"] == "unknown_table" and "kusim run" in answer["error"]
