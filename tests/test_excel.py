"""Loading historical data from Excel or CSV (kusim.excel, ``kusim import-history``)."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openpyxl")
from openpyxl import Workbook  # noqa: E402

from kusim import load_observations  # noqa: E402
from kusim.deck import read_deck  # noqa: E402
from kusim.excel import read_history  # noqa: E402
from kusim.spec import SpecError  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "eclipse"


@pytest.fixture(scope="module")
def model():
    return read_deck(EXAMPLE / "waterflood.DATA").model            # wells P1, P2, I1; dt 10, 600 days


def workbook(tmp_path, rows, sheet="history", name="data.xlsx"):
    book = Workbook()
    ws = book.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    path = tmp_path / name
    book.save(path)
    return path


def table(imported):
    return imported.observations.table.set_index(["vector", "time"])


def error(path, **kwargs):
    with pytest.raises(SpecError) as caught:
        read_history(path, **kwargs)
    return caught.value


# ------------------------------------------------------------------ layouts
def test_by_well_layout_with_units_in_the_headers(tmp_path, model):
    rows = [["Days", "Well", "BHP (bar)", "Oil rate (m3/d)", "Water cut (%)"],
            [5, "P1", 100.0, 100.0, 1.0], [10, "P1", 110.0, 110.0, 3.0],
            [15, "P1", 120.0, 120.0, 5.0], [20, "P1", 120.0, 100.0, 7.0]]
    t = table(read_history(workbook(tmp_path, rows), model=model, sigma={"WBHP": 20, "WOPR": 50, "WWCT": 0.01}))
    assert t.loc[("WBHP:P1", 10.0), "value"] == pytest.approx(105.0 * 14.5037738)               # mean of day 5 and 10, in psi
    assert t.loc[("WOPR:P1", 20.0), "value"] == pytest.approx(110.0 * 6.28981077)               # m3/d -> STB/d
    assert t.loc[("WWCT:P1", 20.0), "value"] == pytest.approx(0.06)                              # percent -> fraction
    assert set(t.index.get_level_values("time")) == {10.0, 20.0}                                 # on the model's time step


def test_align_last_takes_the_last_value_of_the_interval(tmp_path, model):
    rows = [["Days", "Well", "BHP"], [5, "P1", 1000.0], [10, "P1", 1100.0]]
    path = workbook(tmp_path, rows)
    assert table(read_history(path, model=model, align="last")).loc[("WBHP:P1", 10.0), "value"] == 1100.0
    assert table(read_history(path, model=model)).loc[("WBHP:P1", 10.0), "value"] == 1050.0


def test_dates_are_counted_from_start_date_or_the_first_date(tmp_path, model):
    rows = [["Date", "Well", "BHP (psi)"], [datetime(2020, 1, 11), "P1", 1500.0], [datetime(2020, 1, 21), "P1", 1490.0]]
    path = workbook(tmp_path, rows)
    first = read_history(path, model=model)
    assert sorted(first.observations.table.time) == [10.0]               # the first date is day 0 and is dropped
    assert first.report["start_date"] == "2020-01-11" and "time at or before day 0" in first.report["dropped"]
    given = read_history(path, model=model, start_date="2020-01-01")
    assert sorted(given.observations.table.time) == [10.0, 20.0]


def test_wide_layout_with_several_header_styles(tmp_path, model):
    rows = [["Time", "WBHP:P1", "P2 BHP", "oil rate P1", "WWIR_I1", "notes"], [10, 1500, 1400, 2900, 6000, "x"],
            [20, 1490, 1390, 2800, 6000, "y"]]
    imported = read_history(workbook(tmp_path, rows), model=model, layout="wide")
    assert imported.report["layout"] == "wide" and imported.report["columns_ignored"] == ["notes"]
    assert set(imported.observations.table.vector) == {"WBHP:P1", "WBHP:P2", "WOPR:P1", "WWIR:I1"}


def test_long_layout_with_sigma_and_weight(tmp_path, model):
    rows = [["time", "vector", "value", "sigma", "weight"], [10, "WBHP:P1", 1500.0, 15.0, 2.0], [10, "WWCT:P1", 0.05, None, None],
            [20, "WBHP:P1", 1490.0, 20.0, 1.0]]
    imported = read_history(workbook(tmp_path, rows), model=model, sigma={"WWCT": 0.03})
    t = table(imported)
    assert imported.report["layout"] == "long"
    assert t.loc[("WBHP:P1", 10.0), "sigma"] == 15.0 and t.loc[("WBHP:P1", 10.0), "weight"] == 2.0
    assert t.loc[("WWCT:P1", 10.0), "sigma"] == 0.03 and t.loc[("WWCT:P1", 10.0), "weight"] == 1.0   # sheet value, else option


def test_csv_files_are_read_too(tmp_path, model):
    path = tmp_path / "data.csv"
    path.write_text("Days,Well,BHP (psi)\n10,P1,1500\n20,P1,1490\n")
    assert len(read_history(path, model=model).observations.table) == 2


# ------------------------------------------------------------------ wells, sigma, window
def test_well_names_are_mapped(tmp_path, model):
    rows = [["Days", "Well", "BHP"], [10, "PROD-A", 1500.0], [10, "inj-1", 3000.0]]
    path = workbook(tmp_path, rows)
    imported = read_history(path, model=model, wells={"PROD-A": "P1", "INJ-1": "I1"})
    assert set(imported.observations.table.vector) == {"WBHP:P1", "WBHP:I1"}
    problem = error(path, model=model)
    assert problem.code == "unmapped_well" and problem.as_dict()["wells_in_sheet"] == ["PROD-A", "inj-1"]
    assert error(path, model=model, wells={"PROD-A": "P9", "inj-1": "I1"}).code == "unknown_well"


def test_quantities_the_well_does_not_have_are_refused(tmp_path, model):
    rows = [["Days", "Well", "Oil rate"], [10, "I1", 100.0]]
    assert error(workbook(tmp_path, rows), model=model).code == "bad_vector_for_well"


def test_sigma_absolute_relative_and_default(tmp_path, model):
    rows = [["Days", "Well", "BHP", "Oil rate"], [10, "P1", 1500.0, 3000.0], [20, "P1", 1500.0, 10.0]]
    imported = read_history(workbook(tmp_path, rows), model=model, sigma={"bhp": 40, "WOPR": "10%"})
    t = table(imported)
    assert t.loc[("WBHP:P1", 10.0), "sigma"] == 40
    assert t.loc[("WOPR:P1", 10.0), "sigma"] == pytest.approx(300.0)
    assert t.loc[("WOPR:P1", 20.0), "sigma"] == pytest.approx(30.0)               # not 1.0: at least 1 % of the largest value
    assert imported.report["defaults_used"]["WBHP"].startswith("given")
    plain = read_history(workbook(tmp_path, rows, name="b.xlsx"), model=model)
    assert plain.report["defaults_used"]["WBHP"].startswith("default") and "default_sigma" in [w["code"] for w in plain.report["warnings"]]


def test_nothing_is_dropped_silently(tmp_path, model):
    rows = [["Days", "Well", "BHP", "Water cut"], [10, "P1", 1500.0, 0.1], [20, "P1", None, 0.2], [30, "P1", "pending", 0.3],
            [40, "P1", -5.0, 0.4], [650, "P1", 1400.0, 0.5], ["x", "P1", 1400.0, 0.5], [60, "P1", 1300.0, None]]
    imported = read_history(workbook(tmp_path, rows), model=model)
    dropped = imported.report["dropped"]
    assert dropped["blank value"]["count"] == 2 and dropped["blank value"]["first_rows"] == [3, 8]       # Excel row numbers
    assert dropped["value is not a number"]["first_rows"] == [4]
    assert dropped["negative value"]["first_rows"] == [5]
    assert dropped["time is not a number"]["first_rows"] == [7]
    assert any(k.startswith("after the end time") for k in dropped)
    assert len(imported.observations.table) == 6                                                 # what is left, nothing else


def test_vectors_filter(tmp_path, model):
    rows = [["Days", "Well", "BHP", "Water cut"], [10, "P1", 1500.0, 0.1]]
    imported = read_history(workbook(tmp_path, rows), model=model, vectors=["WWCT"])
    assert list(imported.observations.table.vector) == ["WWCT:P1"]
    assert error(workbook(tmp_path, rows, name="b.xlsx"), model=model, vectors=["XYZ"]).code == "bad_value"


# ------------------------------------------------------------------ errors that say what to do
def test_a_water_cut_in_percent_is_caught(tmp_path, model):
    rows = [["Days", "Well", "Water cut"], [10, "P1", 12.0]]
    problem = error(workbook(tmp_path, rows), model=model)
    assert problem.code == "bad_value" and "percent" in problem.as_dict()["hint"]
    assert len(read_history(workbook(tmp_path, rows, name="b.xlsx"), model=model, water_cut_unit="percent").observations.table) == 1


def test_file_and_sheet_errors(tmp_path, model):
    rows = [["Days", "Well", "BHP"], [10, "P1", 1500.0]]
    path = workbook(tmp_path, rows, sheet="Production")
    problem = error(path, model=model, sheet="production ")
    assert problem.code == "unknown_sheet" and problem.as_dict()["allowed"] == ["Production"]
    assert len(read_history(path, model=model, sheet="Production").observations.table) == 1
    assert error(tmp_path / "missing.xlsx").code == "file_not_found"
    (tmp_path / "broken.xlsx").write_text("not a workbook")
    assert error(tmp_path / "broken.xlsx").code == "unreadable_file"
    (tmp_path / "old.xls").write_text("x")
    assert error(tmp_path / "old.xls").code == "unsupported_file"


def test_header_problems(tmp_path, model):
    assert error(workbook(tmp_path, [["Well", "BHP"], ["P1", 1500.0]]), model=model).code == "missing_column"
    assert error(workbook(tmp_path, [["Days", "Well", "Colour"], [10, "P1", "red"]], name="b.xlsx"), model=model).code == "missing_column"
    assert error(workbook(tmp_path, [["Days", "Well", "BHP (furlongs)"], [10, "P1", 1.0]], name="c.xlsx"), model=model).code == "unknown_unit"
    rows = [["time", "vector", "value"], [10, "FOO:P1", 1.0]]
    assert error(workbook(tmp_path, rows, name="d.xlsx"), model=model).code == "unknown_vector"


def test_header_row_can_be_moved(tmp_path, model):
    rows = [["Field data of the test field"], ["Days", "Well", "BHP"], [10, "P1", 1500.0]]
    imported = read_history(workbook(tmp_path, rows), model=model, header_row=2)
    assert len(imported.observations.table) == 1


def test_no_dt_means_times_are_kept_and_the_user_is_told(tmp_path):
    rows = [["Days", "Well", "BHP"], [7, "P1", 1500.0]]
    imported = read_history(workbook(tmp_path, rows))
    assert list(imported.observations.table.time) == [7.0]
    assert "times_not_aligned" in [w["code"] for w in imported.report["warnings"]]


# ------------------------------------------------------------------ controls from measured rates
def test_controls_from_rates_act_from_the_start_of_the_interval(tmp_path, model):
    rows = [["Days", "Well", "Oil rate", "Water rate", "Water injection"],
            [10, "P1", 2900.0, 100.0, None], [20, "P1", 2900.0, 100.0, None], [30, "P1", 2500.0, 100.0, None],
            [10, "I1", None, None, 6000.0], [20, "I1", None, None, 6000.0], [30, "I1", None, None, 7000.0]]
    imported = read_history(workbook(tmp_path, rows), model=model, controls=True)
    events = [(e.time, e.well, e.mode, e.target) for e in imported.schedule.events]
    assert (0.0, "P1", "liquid_rate", 3000.0) in events                    # oil + water; the rate of (0, 10] acts from day 0
    assert (20.0, "P1", "liquid_rate", 2600.0) in events and (20.0, "I1", "water_rate", 7000.0) in events
    assert sum(1 for e in events if e[1] == "P1") == 2                       # the repeated 3000 is not an event
    oil = read_history(workbook(tmp_path, rows, name="b.xlsx"), model=model, controls=True, producer_control="oil_rate")
    assert (20.0, "P1", "oil_rate", 2500.0) in [(e.time, e.well, e.mode, e.target) for e in oil.schedule.events]


def test_controls_need_a_time_step(tmp_path):
    rows = [["Days", "Well", "Oil rate"], [10, "P1", 100.0]]
    assert error(workbook(tmp_path, rows), controls=True).code == "controls_need_dt"


# ------------------------------------------------------------------ the command line
def _cli(*args):
    return subprocess.run([sys.executable, "-m", "kusim", *map(str, args)], capture_output=True, text=True, encoding="utf8")


def test_template_then_import_then_misfit(tmp_path):
    assert json.loads(_cli("history-template", "-o", tmp_path / "h.xlsx").stdout)["ok"]
    _cli("import-deck", EXAMPLE / "waterflood.DATA", "-o", tmp_path / "m.json", "--schedule-out", tmp_path / "s.json")
    done = _cli("import-history", tmp_path / "h.xlsx", "-o", tmp_path / "obs.json", "--model", tmp_path / "m.json",
                "--start-date", "2020-01-01", "--vectors", "WBHP,WWPR", "--sigma", "WBHP=30", "--controls-out", tmp_path / "c.json")
    answer = json.loads(done.stdout)
    assert done.returncode == 0 and answer["ok"] and answer["layout"] == "by_well"
    assert answer["defaults_used"]["WBHP"].startswith("given") and answer["controls_written"].endswith("c.json")
    observations = load_observations(tmp_path / "obs.json")
    assert set(observations.vectors) == {f"WBHP:{w}" for w in ("P1", "P2", "I1")} | {"WWPR:P1", "WWPR:P2"}
    misfit = json.loads(_cli("misfit", tmp_path / "m.json", "--observations", tmp_path / "obs.json", "--schedule",
                             tmp_path / "c.json").stdout)
    assert misfit["ok"] and misfit["n_compared"] == len(observations.table) and misfit["rms"] > 0


def test_well_map_options_and_wells_file(tmp_path):
    rows = [["Days", "Well", "BHP"], [10, "PROD-A", 1500.0], [10, "INJ-1", 3000.0]]
    path = workbook(tmp_path, rows)
    (tmp_path / "wells.json").write_text(json.dumps({"PROD-A": "P1"}))
    done = _cli("import-history", path, "-o", tmp_path / "o.json", "--dt", 10, "--wells-file", tmp_path / "wells.json",
                "--well-map", "INJ-1=I1")
    assert json.loads(done.stdout)["wells"] == {"PROD-A": "P1", "INJ-1": "I1"}


def test_import_history_errors_are_json_with_exit_code_2(tmp_path):
    rows = [["Days", "Well", "BHP"], [10, "PROD-A", 1500.0]]
    done = _cli("import-history", workbook(tmp_path, rows), "-o", tmp_path / "o.json", "--dt", 10)
    answer = json.loads(done.stdout)
    assert done.returncode == 2 and answer["ok"] is False and answer["code"] == "unmapped_well" and "hint" in answer
    bad = json.loads(_cli("import-history", workbook(tmp_path, rows, name="b.xlsx"), "-o", tmp_path / "o.json",
                          "--sigma", "oops").stdout)
    assert bad["ok"] is False and bad["code"] == "bad_value"
    assert not (tmp_path / "o.json").exists()
