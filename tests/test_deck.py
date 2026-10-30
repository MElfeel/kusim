"""Reading an ECLIPSE-style deck (kusim.deck, ``kusim import-deck``)."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kusim import Simulator, load_model, load_schedule
from kusim.deck import read_deck
from kusim.spec import SpecError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "eclipse"


@pytest.fixture
def deck(tmp_path):
    """Write the example deck (with its include file) into a folder, changing some text; returns the path."""
    def make(replace=(), name="case.DATA", append_before_end=""):
        text = (EXAMPLE / "waterflood.DATA").read_text()
        for old, new in replace:
            assert old in text, old
            text = text.replace(old, new, 1)
        if append_before_end:
            text = text.replace("END\n", append_before_end + "\nEND\n")
        shutil.copy(EXAMPLE / "waterflood_props.INC", tmp_path / "waterflood_props.INC")
        path = tmp_path / name
        path.write_text(text)
        return path
    return make


def _error(path, **kwargs):
    with pytest.raises(SpecError) as caught:
        read_deck(path, **kwargs)
    return caught.value


# ------------------------------------------------------------------ the example deck
def test_the_example_deck_becomes_the_expected_model():
    imported = read_deck(EXAMPLE / "waterflood.DATA")
    m = imported.model
    assert (m.grid.nx, m.grid.ny, m.grid.nz) == (8, 6, 3)
    assert m.dt == 10.0 and m.t_end == 600.0
    assert imported.wells == {"I1": "I1", "P1": "P1", "P2": "P2"}
    assert [w.blocks for w in m.producers] == [[8, 56, 104], [48, 96, 144]]          # (8,1,1-3) and (8,6,1-3)
    assert m.injectors[0].blocks == [17, 65, 113] and m.injectors[0].target == 6000.0
    assert m.producers[0].mode == "liquid_rate" and m.producers[0].min_bhp == 1500.0
    assert m.producers[0].radius == pytest.approx(0.25)
    assert [(e.time, e.well, e.mode, e.target) for e in imported.schedule.events] == [(400.0, "I1", "water_rate", 7800.0)]
    assert m.initial.datum_depth == 7600.0 and m.initial.datum_pressure == 3500.0 and m.initial.woc_depth == 7700.0
    assert m.gamma_o == pytest.approx(53.0 / 62.4) and m.gamma_w == 1.0
    assert m.swc == 0.2 and m.sor == pytest.approx(0.2)
    assert m.rock_compressibility == 4e-6
    assert np.allclose(m.kx[:48], 120) and np.allclose(m.kx[48:96], 600) and np.allclose(m.kz[48:96], 60)   # COPY + MULTIPLY
    assert np.allclose(m.ky, m.kx)
    assert {i["keyword"] for i in imported.ignored} >= {"TITLE", "WELLDIMS", "SUMMARY"}      # accepted, no effect
    assert imported.warnings == []


def test_the_initial_pressure_is_set_at_the_datum_depth():
    m = read_deck(EXAMPLE / "waterflood.DATA").model
    p = Simulator(m).p_o
    depth = m.grid.depth
    assert np.allclose(p, 3500.0 + 0.433 * m.gamma_o * (depth - 7600.0))


def test_pvtw_and_pvdo_make_the_fluid_table():
    m = read_deck(EXAMPLE / "waterflood.DATA").model
    assert list(m.pvt.pressure) == [1000, 2000, 3000, 4000, 5000]
    assert m.pvt.bo[0] == 1.3 and m.pvt.mu_o[0] == 1.5
    x = 3e-6 * (m.pvt.pressure - 3500.0)
    assert np.allclose(m.pvt.bw, 1.02 / (1 + x + x * x / 2))             # PVTW: Bw falls with pressure
    assert np.allclose(m.pvt.mu_w, 0.5 / (1 + (3e-6) * (m.pvt.pressure - 3500.0) * (1 + 0.5 * 3e-6 * (m.pvt.pressure - 3500.0))), rtol=1e-3)


def test_the_imported_model_runs_healthy():
    imported = read_deck(EXAMPLE / "waterflood.DATA")
    from kusim.schedule import advance
    sim = Simulator(imported.model)
    advance(sim, imported.model.t_end, imported.schedule)
    from kusim.simulator import diagnostics_healthy
    assert diagnostics_healthy(sim.diagnostics())
    assert sim.time == 600.0


# ------------------------------------------------------------------ keywords that change the grid and the arrays
def test_equals_multiply_add_copy_with_a_box(deck):
    path = deck([("PORO\n 144*0.2 /", "PORO\n 144*0.2 /\nEQUALS\n PORO 0.3 1 8 1 6 3 3 /\n/\nADD\n PORO 0.01 1 2 1 1 1 1 /\n/\n"
                  "MULTIPLY\n PORO 0.5 1 1 1 1 1 1 /\n/")])
    poro = read_deck(path).model.porosity.reshape(3, 6, 8)
    assert np.allclose(poro[2], 0.3) and np.allclose(poro[1], 0.2)
    assert poro[0, 0, 0] == pytest.approx(0.105) and poro[0, 0, 1] == pytest.approx(0.21) and poro[0, 1, 0] == 0.2


def test_dx_dy_dz_arrays_must_depend_only_on_their_own_index(deck):
    path = deck([("DXV\n 8*300 /", "DX\n 144*300 /")])
    assert read_deck(path).model.grid.dx[0] == 300
    bad = deck([("DXV\n 8*300 /", "DX\n 143*300 400 /")], name="bad.DATA")
    error = _error(bad)
    assert error.code == "unsupported_geometry" and "DX" in str(error)


def test_tops_may_vary_by_column(deck):
    path = deck([("TOPS\n 48*7500 /", "TOPS\n 24*7500 24*7510 /")])
    top = read_deck(path).model.grid.depth[:48] - 15.0
    assert np.allclose(top[:24], 7500.0) and np.allclose(top[24:], 7510.0)


# ------------------------------------------------------------------ strictness: what is refused, with where and why
@pytest.mark.parametrize("old,new,code,word", [
    ("FIELD\n", "METRIC\n", "unsupported_units", "METRIC"),
    ("FIELD\n", "", "units_not_declared", "FIELD"),
    ("OIL\n", "OIL\nGAS\n", "unsupported_feature", "GAS"),
    ("PROPS\n", "PROPS\nPVTO\n 1 1 1 1 /\n/\n", "unsupported_feature", "PVTO"),
    ("GRID\n", "GRID\nCOORD\n 1 /\n", "unsupported_geometry", "COORD"),
    ("PROPS\n", "PROPS\nSWL\n 144*0.2 /\n", "unsupported_feature", "SWL"),
    ("PROPS\n", "PROPS\nFROBNICATE\n 1 /\n", "unsupported_keyword", "FROBNICATE"),
    ("EQUIL\n 7600 3500 7700 0 /", "EQUIL\n 7600 3500 7700 5 /", "unsupported_value", "capillary"),
    ("PROPS\n", "PROPS\nNTG\n 144*0.8 /\n", "unsupported_feature", "NTG"),
    ("'I1' 'WATER' 'OPEN' 'RATE' 6000 /", "'I1' 'GAS' 'OPEN' 'RATE' 6000 /", "unsupported_feature", "GAS"),
    ("'LRAT' 3* 3000 1* 1500 /\n 'P2'", "'WRAT' 3* 3000 1* 1500 /\n 'P2'", "unsupported_value", "WRAT"),
    ("COMPDAT\n", "COMPDAT\n 'P1' 1* 1* 1 1 'OPEN' 1* 1* 0.5 1* 1* 1* 'X' /\n", "unsupported_feature", "direction"),
    ("TSTEP\n 40*10 /", "WCONPROD\n 'PX' 'OPEN' 'BHP' 5* 1000 /\n/\nTSTEP\n 40*10 /", "unknown_well", "PX"),
])
def test_unsupported_input_stops_the_import_with_a_located_error(deck, old, new, code, word):
    path = deck([(old, new)])
    error = _error(path)
    assert error.code == code, error
    assert word in str(error) and re.match(r"^(case\.DATA|waterflood_props\.INC):\d+$", error.path) or error.code == "units_not_declared"


def test_the_error_names_the_file_and_line(deck):
    path = deck([("PROPS\n", "PROPS\nPVTO\n 1 1 1 1 /\n/\n")])
    error = _error(path)
    line = (path.read_text().splitlines().index("PVTO") + 1)
    assert error.path == f"case.DATA:{line}" and "hint" in error.as_dict()


def test_an_error_inside_an_included_file_names_that_file(deck, tmp_path):
    path = deck()
    inc = tmp_path / "waterflood_props.INC"
    inc.write_text(inc.read_text().replace("PVTW", "PVTO"))
    error = _error(path)
    assert error.path.startswith("waterflood_props.INC:")


def test_skip_unknown_turns_errors_into_warnings(deck):
    path = deck([("PROPS\n", "PROPS\nFROBNICATE\n 1 2 3 /\n")])
    imported = read_deck(path, skip_unknown=True)
    assert [w["code"] for w in imported.warnings] == ["unsupported_keyword"]
    assert any(i["keyword"] == "FROBNICATE" for i in imported.ignored)
    assert imported.model.grid.nx == 8


def test_missing_pieces_are_named(deck):
    assert _error(deck([("ROCK\n 3500 4E-6 /\n", "")])).code == "missing_keyword"
    assert _error(deck([("SOLUTION\n", "SOLUTION\nOIL\n"), ("EQUIL\n 7600 3500 7700 0 /", "")])).code == "missing_keyword"
    assert _error(deck([("TSTEP\n 40*10 /", ""), ("TSTEP\n 20*10 /", "")])).code == "no_time_steps"


def test_two_regions_are_refused(deck):
    path = deck([("ROCK\n 3500 4E-6 /", "ROCK\n 3500 4E-6 /\n 3500 5E-6 /")])
    assert _error(path).code == "unsupported_feature"


def test_all_ones_region_arrays_are_accepted(deck):
    path = deck([("PROPS\n", "PROPS\nSATNUM\n 144*1 /\n")])
    read_deck(path)
    assert _error(deck([("PROPS\n", "PROPS\nSATNUM\n 143*1 2 /\n")], name="r.DATA")).code == "unsupported_feature"


# ------------------------------------------------------------------ the schedule and the time axis
def test_dt_is_the_common_divisor_of_the_intervals_unless_given(deck):
    path = deck([("TSTEP\n 40*10 /", "TSTEP\n 4*30 10*15 /"), ("TSTEP\n 20*10 /", "")])
    assert read_deck(path).model.dt == 15.0 and read_deck(path).model.t_end == 270.0
    assert read_deck(path, dt=5.0).model.dt == 5.0
    error = _error(path, dt=4.0)
    assert error.code == "dt_not_commensurate" and "times" in error.as_dict()


def test_dates_use_start(deck):
    path = deck([("TSTEP\n 40*10 /", "DATES\n 1 FEB 2020 /\n 1 MAR 2020 /\n/"), ("TSTEP\n 20*10 /", "")])
    imported = read_deck(path)
    assert imported.model.t_end == 60.0                              # 31 + 29 days
    no_start = deck([("TSTEP\n 40*10 /", "DATES\n 1 FEB 2020 /\n/"), ("TSTEP\n 20*10 /", ""), ("START\n 1 JAN 2020 /", "")],
                    name="n.DATA")
    assert _error(no_start).code == "missing_keyword"


def test_wconhist_weltarg_welopen_make_events(deck):
    extra = ("WCONHIST\n 'P1' 'OPEN' 'ORAT' 2800 100 /\n/\nTSTEP\n 10 /\n"
             "WELTARG\n 'P1' 'ORAT' 2500 /\n/\nTSTEP\n 10 /\nWELOPEN\n 'P2' 'SHUT' /\n/\nTSTEP\n 10 /\n"
             "WELOPEN\n 'P2' 'OPEN' /\n/\nTSTEP\n 10 /")
    path = deck([("TSTEP\n 20*10 /\nEND", "TSTEP\n 20*10 /\n" + extra + "\nEND")])
    events = [(e.time, e.well, e.mode, e.target) for e in read_deck(path).schedule.events]
    assert (600.0, "P1", "oil_rate", 2800.0) in events                       # WCONHIST: oil rate control
    assert (610.0, "P1", "oil_rate", 2500.0) in events                       # WELTARG changes the target of the active mode
    assert (620.0, "P2", "liquid_rate", 0.0) in events                       # SHUT is a zero rate
    assert (630.0, "P2", "liquid_rate", 3000.0) in events                    # OPEN gives the control back


def test_controls_before_the_first_step_are_the_initial_controls(deck):
    path = deck([("'I1' 'WATER' 'OPEN' 'RATE' 6000 /", "'I1' 'WATER' 'OPEN' 'RATE' 6000 /\n 'I1' 'WATER' 'OPEN' 'RATE' 5000 /")])
    assert read_deck(path).model.injectors[0].target == 5000.0                  # the last record before time passes


def test_a_well_with_a_late_first_control_is_shut_until_then(deck):
    path = deck([("WCONPROD\n", "WCONPROD\n"), ("TSTEP\n 40*10 /", "TSTEP\n 5*10 /\nWCONPROD\n 'P2' 'OPEN' 'BHP' 5* 2000 /\n/\nTSTEP\n 35*10 /"),
                 (" 'P2' 'OPEN' 'LRAT' 3* 3000 1* 1500 /\n", "")])
    imported = read_deck(path)
    assert imported.model.producers[1].mode == "liquid_rate" and imported.model.producers[1].target == 0.0
    assert [w["code"] for w in imported.warnings] == ["well_starts_late"]
    assert any(e.well == "P2" and e.time == 50.0 and e.mode == "bhp" for e in imported.schedule.events)


def test_pvcdo_makes_a_table(deck):
    path = deck([("INCLUDE\n 'waterflood_props.INC' /", "INCLUDE\n 'props_swof_only.INC' /\nPVCDO\n 3500 1.27 1E-5 1.5 0 /\nPVTW\n 3500 1.02 3E-6 0.5 0 /")])
    inc = path.parent / "props_swof_only.INC"
    text = (EXAMPLE / "waterflood_props.INC").read_text()
    inc.write_text(text[text.index("SWOF"):])
    imported = read_deck(path)
    assert len(imported.model.pvt.pressure) == 16 and "pressure_grid_chosen" in [w["code"] for w in imported.warnings]
    assert imported.model.pvt.bo[0] > imported.model.pvt.bo[-1]
    assert len(read_deck(path, pvt_pressures=[1000, 3500, 6000]).model.pvt.pressure) == 3


def test_swfn_and_sof2_make_the_same_table_as_swof(deck, tmp_path):
    (tmp_path / "x").mkdir()
    path = deck([("INCLUDE\n 'waterflood_props.INC' /", "INCLUDE\n 'p2.INC' /")])
    inc = (EXAMPLE / "waterflood_props.INC").read_text()
    rows = [[float(x) for x in line.split()[:4]] for line in inc[inc.index("SWOF"):].splitlines()[2:11]]
    swfn = "SWFN\n" + "\n".join(f"{r[0]} {r[1]} {r[3]}" for r in rows) + " /\n"
    sof2 = "SOF2\n" + "\n".join(f"{1 - r[0]:.2f} {r[2]}" for r in reversed(rows)) + " /\n"
    (tmp_path / "p2.INC").write_text(inc[:inc.index("SWOF")] + swfn + sof2)
    a, b = read_deck(path).model, read_deck(EXAMPLE / "waterflood.DATA").model
    assert np.allclose(a.rock_fluid.kro, b.rock_fluid.kro) and np.allclose(a.rock_fluid.pcow, b.rock_fluid.pcow)


# ------------------------------------------------------------------ the command line
def _cli(*args):
    return subprocess.run([sys.executable, "-m", "kusim", *map(str, args)], capture_output=True, text=True, encoding="utf8")


def test_import_deck_command_writes_model_schedule_and_wells(tmp_path):
    done = _cli("import-deck", EXAMPLE / "waterflood.DATA", "-o", tmp_path / "m.json", "--schedule-out", tmp_path / "s.json",
                "--wells-out", tmp_path / "w.json")
    answer = json.loads(done.stdout)
    assert done.returncode == 0 and answer["ok"] and answer["runnable"] and answer["n_events"] == 1
    assert answer["wells"] == {"I1": "I1", "P1": "P1", "P2": "P2"} and answer["mapping"]
    model, schedule = load_model(tmp_path / "m.json"), load_schedule(tmp_path / "s.json")
    assert model.initial.datum_depth == 7600.0 and len(schedule) == 1
    assert json.loads((tmp_path / "w.json").read_text()) == answer["wells"]
    assert _cli("validate", tmp_path / "m.json", "--schedule", tmp_path / "s.json").returncode == 0


def test_import_deck_warns_when_the_schedule_is_not_written(tmp_path):
    answer = json.loads(_cli("import-deck", EXAMPLE / "waterflood.DATA", "-o", tmp_path / "m.json").stdout)
    assert [w["code"] for w in answer["warnings"]] == ["schedule_not_written"]


def test_import_deck_errors_are_json_with_exit_code_2(tmp_path, deck):
    path = deck([("PROPS\n", "PROPS\nPVTO\n 1 1 1 1 /\n/\n")])
    done = _cli("import-deck", path, "-o", tmp_path / "m.json")
    answer = json.loads(done.stdout)
    assert done.returncode == 2 and answer["ok"] is False and answer["code"] == "unsupported_feature"
    assert answer["path"].startswith("case.DATA:") and "hint" in answer
    assert not (tmp_path / "m.json").exists()
    missing = json.loads(_cli("import-deck", tmp_path / "nope.DATA", "-o", tmp_path / "m.json").stdout)
    assert missing["ok"] is False


def test_import_deck_flags_a_time_step_that_is_too_large(tmp_path, deck):
    path = deck([("TSTEP\n 40*10 /", "TSTEP\n 10*60 /"), ("TSTEP\n 20*10 /", "TSTEP\n 5*60 /")])
    answer = json.loads(_cli("import-deck", path, "-o", tmp_path / "m.json", "--schedule-out", tmp_path / "s.json").stdout)
    assert answer["ok"] and "dt_too_large" in [f["code"] for f in answer["checks"]["findings"]]
