"""Checks before a run: size, time, output, stability (kusim.checks) and how the command line uses them."""

import json
import math

import numpy as np
import pytest

from conftest import LEGACY_INPUT
from kusim import Simulator, read_legacy_input, save_model
from kusim.simulator import SATURATION_CLIP_TOLERANCE, diagnostics_healthy
from kusim.cases import layered_waterflood
from kusim.checks import COURANT_WARN, check_model, output_megabytes, probe
from kusim.cli import main
from kusim.history import Observations
from kusim.spec import save_observations


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


def _codes(check):
    return {f["code"] for f in check["findings"]}


def _layered(dt, nx=10, ny=10, nz=3, steps=40):
    return layered_waterflood(nx=nx, ny=ny, nz=nz, days=steps * dt).with_changes(dt=float(dt))


# ---------------------------------------------------------------------- the stability heuristic
def test_throughput_is_nan_before_the_first_step_and_grows_with_dt():
    sim = Simulator(_layered(5.0))
    assert math.isnan(sim.well_block_throughput())
    values = []
    for dt in (1.0, 5.0, 20.0):
        s = Simulator(_layered(dt))
        s.step()
        values.append(s.well_block_throughput())
    assert values[0] < values[1] < values[2] and values[2] == pytest.approx(4.08, abs=0.01)


def _clipped(model):
    sim = Simulator(model)
    sim.run()
    return sim.diagnostics()["max_saturation_clip"]


def _rate_controlled(dt):
    model = layered_waterflood(nx=10, ny=10, nz=3, days=400.0)
    return model.with_changes({"producers[0].mode": "liquid_rate", "producers[0].target": 150.0}).with_changes(
        dt=dt, t_end=40 * dt)


def _thesis(dt):
    return read_legacy_input(LEGACY_INPUT).with_changes(dt=dt, t_end=40 * dt)


@pytest.mark.parametrize("make, dt", [(_layered, 1.0), (_layered, 10.0), (_layered, 20.0), (_layered, 40.0),
                                      (_rate_controlled, 5.0), (_rate_controlled, 10.0), (_rate_controlled, 20.0),
                                      (_rate_controlled, 40.0), (_thesis, 0.25), (_thesis, 0.5), (_thesis, 1.0)])
def test_the_check_agrees_with_the_whole_run(make, dt):
    """The claim of the module docstring: a ``dt_too_large`` warning if and only if the run moves saturations by tenths.
    Three families: a layered waterflood with a pressure-controlled producer, the same with a rate-controlled one (unstable
    from a throughput of about 1.4 instead of 3.5) and the thesis input (whose instability starts later)."""
    model = make(dt)
    flagged = "dt_too_large" in _codes(check_model(model))
    assert flagged == (_clipped(model) > SATURATION_CLIP_TOLERANCE)


def test_the_throughput_alone_would_miss_a_rate_controlled_case():
    """Why the probe itself is a signal: at dt = 20 the throughput is 2.4, below its threshold, yet saturations move by 0.55."""
    model = _rate_controlled(20.0)
    p = probe(model, 3)
    assert p["well_block_throughput"] < COURANT_WARN and p["diagnostics"]["max_saturation_clip"] > SATURATION_CLIP_TOLERANCE
    assert check_model(model)["stability"]["likely_stable"] is False


# ---------------------------------------------------------------------- findings
def test_a_stable_model_has_no_findings():
    check = check_model(_layered(5.0))
    assert check["runnable"] and check["findings"] == [] and check["stability"]["likely_stable"]
    assert check["estimate"]["seconds"] > 0 and check["estimate"]["n_steps"] == 40


def test_dt_too_large_suggests_a_fraction_of_dt():
    model = _layered(40.0)
    check = check_model(model)
    (f,) = [f for f in check["findings"] if f["code"] == "dt_too_large"]
    assert check["runnable"] and f["severity"] == "warning" and not check["stability"]["likely_stable"]
    k = model.dt / f["suggested_dt"]
    assert k == round(k) and k >= 2                                   # existing times stay on the new time grid
    better = check_model(model.with_changes(dt=f["suggested_dt"]))
    assert "dt_too_large" not in _codes(better)                        # and the suggestion really passes the check


def test_size_time_and_output_limits_are_errors():
    model = _layered(5.0)
    assert "too_large" in _codes(check_model(model, max_cells=100))
    assert check_model(model, max_cells=100)["estimate"] is None             # no probe on a model that is refused
    check = check_model(model, max_seconds=1e-9)
    assert not check["runnable"] and "too_slow" in _codes(check)
    check = check_model(model, fields="all", max_output_mb=1e-4)
    assert not check["runnable"] and "output_too_large" in _codes(check)
    assert check_model(model, max_cells=0, max_seconds=0, max_output_mb=0)["runnable"]        # 0 switches a limit off


def test_repeats_scale_the_time_estimate():
    one = check_model(_layered(5.0), repeats=1)["estimate"]
    many = check_model(_layered(5.0), repeats=50)["estimate"]
    assert many["seconds"] == pytest.approx(50 * many["seconds_one_run"]) and many["repeats"] == 50
    assert one["repeats"] == 1


def test_pressure_outside_the_pvt_table_is_found_before_running():
    model = layered_waterflood(nx=3, ny=1, nz=1, days=5.0, producer_bhp=200.0)
    check = check_model(model, probe_steps=0)
    (f,) = [f for f in check["findings"] if f["code"] == "pressure_outside_pvt_range"]
    assert f["well"] == "P1" and f["severity"] == "warning" and check["runnable"]


def test_the_gauss_solver_is_flagged_for_big_models():
    assert "slow_solver" in _codes(check_model(_layered(5.0, nz=5), linear_solver="gauss", probe_steps=0))
    small = layered_waterflood(nx=3, ny=3, nz=2, days=5.0)
    assert "slow_solver" not in _codes(check_model(small, linear_solver="gauss", probe_steps=0))


def test_the_output_size_estimate_is_the_right_order(tmp_path):
    model = layered_waterflood(nx=6, ny=5, nz=2, days=30.0)
    sim = Simulator(model)
    results = sim.run()
    for fields in ("none", "last", "all"):
        results.to_json(tmp_path / f"{fields}.json", fields=fields)
        real = (tmp_path / f"{fields}.json").stat().st_size / 1e6
        estimate = output_megabytes(model, fields)
        assert 0.4 < estimate / real < 2.5, (fields, estimate, real)


# ---------------------------------------------------------------------- the command line
@pytest.fixture
def files(tmp_path):
    save_model(_layered(5.0, nx=4, ny=3, nz=2, steps=6), tmp_path / "m.json")
    save_observations(Observations([{"time": 10.0, "vector": "WBHP:I1", "value": 3000.0, "sigma": 20.0}]), tmp_path / "o.json")
    (tmp_path / "p.json").write_text(json.dumps({"parameters": [{"type": "field", "field": "kx"}]}))
    (tmp_path / "prior.json").write_text(json.dumps({"parameters": {"log_kx": {"std": 0.3}}}))
    return tmp_path


def test_validate_reports_runnable_and_checks(files, capsys):
    code, out = _cli(capsys, "validate", files / "m.json")
    assert code == 0 and out["runnable"] is True and out["checks"]["findings"] == []
    assert out["checks"]["estimate"]["n_steps"] == 6 and out["checks"]["stability"]["likely_stable"]
    code, out = _cli(capsys, "validate", files / "m.json", "--max-cells", 5)
    assert code == 0 and out["runnable"] is False                           # the file is valid; the run would be refused
    assert out["checks"]["findings"][0]["code"] == "too_large"
    code, out = _cli(capsys, "validate", files / "m.json", "--no-probe")
    assert out["checks"]["stability"] is None and out["checks"]["estimate"] is None


def test_validate_warns_about_dt(tmp_path, capsys):
    save_model(_layered(40.0), tmp_path / "big_dt.json")
    code, out = _cli(capsys, "validate", tmp_path / "big_dt.json")
    assert code == 0 and out["runnable"] is True
    (f,) = out["checks"]["findings"]
    assert f["code"] == "dt_too_large" and f["suggested_dt"] < 40.0


def test_run_refuses_and_says_which_flag_to_raise(files, capsys):
    code, out = _cli(capsys, "run", files / "m.json", "--max-cells", 5)
    assert code == 2 and out["ok"] is False and out["code"] == "too_large"
    assert out["findings"][0]["flag"] == "--max-cells" and out["findings"][0]["n_cells"] == 24
    code, out = _cli(capsys, "run", files / "m.json", "--max-seconds", 1e-9)
    assert code == 2 and out["code"] == "too_slow" and out["estimate"]["seconds"] > 0
    code, out = _cli(capsys, "run", files / "m.json", "--fields", "all", "--max-output-mb", 1e-4)
    assert code == 2 and out["code"] == "output_too_large"
    code, out = _cli(capsys, "run", files / "m.json", "--max-cells", 0, "--max-seconds", 0)       # limits off
    assert code == 0 and out["ok"]


def test_run_answer_carries_the_checks_and_the_time(files, capsys):
    code, out = _cli(capsys, "run", files / "m.json")
    assert code == 0 and out["checks"]["findings"] == [] and out["checks"]["estimate"]["n_steps"] == 6
    assert out["elapsed_seconds"] >= 0
    save_model(_layered(40.0), files / "big_dt.json")
    code, out = _cli(capsys, "run", files / "big_dt.json")
    assert code == 0 and out["checks"]["findings"][0]["code"] == "dt_too_large"      # a warning does not stop the run


def test_staged_run_estimates_only_the_remaining_steps(files, capsys):
    _cli(capsys, "run", files / "m.json", "--until", 20, "--state-out", files / "s.json")
    code, out = _cli(capsys, "run", files / "m.json", "--restart", files / "s.json")
    assert out["checks"]["estimate"]["n_steps"] == 2                          # days 20 to 30 in steps of 5


def test_other_commands_refuse_too(files, capsys):
    base = {"misfit": ("misfit", files / "m.json", "--observations", files / "o.json"),
            "gradient": ("gradient", files / "m.json", "--observations", files / "o.json", "--parameters", files / "p.json"),
            "ensemble": ("ensemble", files / "m.json", "--method", "esmda", "--observations", files / "o.json",
                         "--parameters", files / "p.json", "--prior", files / "prior.json")}
    for name, argv in base.items():
        code, out = _cli(capsys, *argv, "--max-cells", 5)
        assert code == 2 and out["code"] == "too_large", name
    code, out = _cli(capsys, *base["misfit"])
    assert code == 0 and "checks" in out
    code, out = _cli(capsys, *base["gradient"])
    assert code == 0 and out["checks"]["estimate"]["repeats"] == 3
    code, out = _cli(capsys, *base["ensemble"], "--members", 10, "--workers", 2, "--max-seconds", 1e-9)
    assert code == 2 and out["code"] == "too_slow" and out["estimate"]["repeats"] == 5 * 5   # 5 batches x (4 passes + 1)


# ---------------------------------------------------------------------- what counts as trouble
def test_healthy_ignores_harmless_clipping_but_not_an_unstable_step():
    quiet = dict(outside_pvt_range=False, stop_reason=None, n_backflow_well_steps=0, n_saturation_clipped=51,
                 max_saturation_clip=1.3e-4)
    assert diagnostics_healthy(quiet)                                           # oil expanding at connate: harmless
    assert not diagnostics_healthy(dict(quiet, max_saturation_clip=0.48))      # a saturation moved by half: unstable
    assert not diagnostics_healthy(dict(quiet, outside_pvt_range=True))
    assert not diagnostics_healthy(dict(quiet, n_backflow_well_steps=1))
    assert not diagnostics_healthy(dict(quiet, stop_reason="P1: oil rate fell"))


def test_the_thesis_case_moves_saturations_at_a_large_dt_only():
    """(The thesis input also starts above its PVT table, which is a separate sign of trouble.)"""
    for dt, material in ((0.25, False), (0.5, True)):
        sim = Simulator(read_legacy_input(LEGACY_INPUT).with_changes(dt=dt, t_end=40 * dt))
        sim.run()
        assert (sim.diagnostics()["max_saturation_clip"] > SATURATION_CLIP_TOLERANCE) is material, dt
