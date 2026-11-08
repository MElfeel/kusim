"""kusim refine, compare and describe, and the well-name paths and warnings of edit (the agent-facing helpers)."""

import json

import numpy as np
import pytest

from kusim import load_model, save_model
from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.refine import refine_model


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def model():
    return layered_waterflood(nx=4, ny=3, nz=2, days=10.0)


def test_refine_keeps_the_reservoir_and_moves_the_wells(model):
    fine, info = refine_model(model, 2, dt="keep")
    g, f = model.grid, fine.grid
    assert (f.nx, f.ny, f.nz) == (8, 6, 4) and f.n_cells == 8 * g.n_cells
    assert np.isclose(f.volume.sum(), g.volume.sum())
    assert np.isclose((fine.porosity * f.volume).sum(), (model.porosity * g.volume).sum())
    for old, new in zip(model.producers + model.injectors, fine.producers + fine.injectors):
        assert len(new.blocks) == 2 * len(old.blocks)                      # all children of each old layer
        ox, oy = g.block_ijk(old.blocks[0])[:2]
        columns = {f.block_ijk(b)[:2] for b in new.blocks}
        assert len(columns) == 1                                           # one child column
        (i, j), = columns
        assert (i - 1) // 2 + 1 == ox and (j - 1) // 2 + 1 == oy           # inside the old column
    assert fine.dt == model.dt and fine.provenance[-1].startswith("refined")


def test_refine_copies_per_block_properties(model):
    kx = np.arange(1, model.grid.n_cells + 1, dtype=float)
    fine, _ = refine_model(model.with_changes(kx=kx), 2, 1, 3, dt="keep")
    assert fine.grid.n_cells == 6 * model.grid.n_cells
    assert sorted(set(fine.kx)) == sorted(set(kx)) and np.isclose(fine.kx.sum(), 6 * kx.sum())


def test_refine_by_one_is_the_same_model(model):
    same, _ = refine_model(model, 1, dt="keep")
    assert same.grid.n_cells == model.grid.n_cells and [w.blocks for w in same.producers] == [w.blocks for w in model.producers]
    assert np.allclose(same.kx, model.kx) and np.allclose(same.grid.volume, model.grid.volume)


def test_refine_rejects_bad_factors(model):
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            refine_model(model, bad)


def test_the_cli_refines_runs_and_compares(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=20.0), tmp_path / "m.json")
    code, answer = _cli(capsys, "refine", tmp_path / "m.json", "--factor", 2, "-o", tmp_path / "fine.json")
    assert code == 0 and answer["new_n_cells"] == 8 * answer["old_n_cells"] and "dt_decision" in answer and set(answer["wells"]) == {"P1", "I1"}
    assert answer["dt"] <= 1.0 and load_model(tmp_path / "fine.json").grid.n_cells == answer["new_n_cells"]
    for name in ("m", "fine"):
        assert _cli(capsys, "run", tmp_path / f"{name}.json", "--out", tmp_path / name)[1]["healthy"]
    code, cmp = _cli(capsys, "compare", tmp_path / "m", tmp_path / "fine")
    assert code == 0 and cmp["same_numbers"] is False and any("different size" in n for n in cmp["notes"])
    assert cmp["wells"]["P1"]["cum_oil"]["a"] > 0 and "change_percent" in cmp["wells"]["P1"]["cum_oil"]


def test_refine_limits_and_errors(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=5.0), tmp_path / "m.json")
    code, answer = _cli(capsys, "refine", tmp_path / "m.json", "--factor", 2, "--max-cells", 100, "-o", tmp_path / "f.json")
    assert code == 2 and answer["code"] == "too_large" and not (tmp_path / "f.json").exists()
    code, answer = _cli(capsys, "refine", tmp_path / "m.json", "--factor", 0, "-o", tmp_path / "f.json")
    assert code == 2 and answer["code"] == "bad_argument"
    code, answer = _cli(capsys, "refine", tmp_path / "m.json", "--dt", "soon", "-o", tmp_path / "f.json")
    assert code == 2 and answer["code"] == "bad_argument"
    code, answer = _cli(capsys, "compare", tmp_path / "nothing", tmp_path / "m.json")
    assert code == 2 and answer["code"] == "file_not_found"


def test_compare_says_when_the_numbers_are_identical(tmp_path, capsys):
    """The failure of the agent exercise: min_bhp on a producer that is already on 'bhp' changes nothing."""
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=10.0), tmp_path / "m.json")
    code, answer = _cli(capsys, "edit", tmp_path / "m.json", "--set", "P1.min_bhp=2000", "-o", tmp_path / "wrong.json")
    assert code == 0 and "min_bhp" in answer["warnings"][0] and "target" in answer["warnings"][0]
    _cli(capsys, "edit", tmp_path / "m.json", "--set", "P1.target=2000", "-o", tmp_path / "right.json")
    for name in ("m", "wrong", "right"):
        _cli(capsys, "run", tmp_path / f"{name}.json", "--out", tmp_path / f"run_{name}")
    same = _cli(capsys, "compare", tmp_path / "run_m", tmp_path / "run_wrong")[1]
    assert same["same_numbers"] is True and any("identical" in n for n in same["notes"])
    different = _cli(capsys, "compare", tmp_path / "run_m", tmp_path / "run_right")[1]
    assert different["same_numbers"] is False and different["wells"]["P1"]["bhp"]["b"] == 2000.0
    assert load_model(tmp_path / "right.json").producers[0].target == 2000.0


def test_well_names_are_paths_and_describe_gives_units(tmp_path, capsys):
    save_model(layered_waterflood(nx=4, ny=3, nz=2, days=5.0), tmp_path / "m.json")
    code, answer = _cli(capsys, "edit", tmp_path / "m.json", "--set", "I1.target=120", "-o", tmp_path / "n.json")
    assert code == 0 and load_model(tmp_path / "n.json").injectors[0].target == 120.0
    code, answer = _cli(capsys, "edit", tmp_path / "m.json", "--set", "P3.target=1", "-o", tmp_path / "n.json")
    assert code == 2 and answer["code"] == "bad_path"
    code, d = _cli(capsys, "describe", tmp_path / "m.json")
    p1, i1 = d["producers"][0], d["injectors"][0]
    assert p1["mode"] == "bhp" and p1["target_unit"] == "psi" and i1["target_unit"] == "STB/day"
    assert len(p1["ijk"]) == len(p1["blocks"]) and d["size_ft"]["x"] > 0 and "target" in d["how_to_change_a_well"]
