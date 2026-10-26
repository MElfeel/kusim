"""History-matching tools: summary vectors, running members, observations and misfit."""

import numpy as np
import pandas as pd
import pytest

from kusim import Injector, Producer, Simulator
from kusim.history import Observations, data_matrix, misfit, objective, rms, run_ensemble, run_to, summary_vectors
from kusim.parameters import FieldParameter, ParameterSet
from kusim.schedule import ControlEvent, Schedule

from conftest import make_model


def _model(**kw):
    defaults = dict(nx=6, ny=1, nz=1, dt=2.0, t_end=60.0, pc=True,
                    injectors=[Injector(1, "water_rate", 60.0)], producers=[Producer(6, "liquid_rate", 70.0, min_bhp=1000.0)])
    defaults.update(kw)
    return make_model(**defaults)


# ------------------------------------------------------------------ summary vectors
def test_summary_vectors_agree_with_the_results_table():
    model = _model(t_end=40.0)
    sim = Simulator(model)
    res = sim.run()
    v = summary_vectors(sim)
    last_p = res.wells[res.wells.well == "P1"].iloc[-1]
    last_i = res.wells[res.wells.well == "I1"].iloc[-1]
    assert v["WBHP:P1"] == pytest.approx(last_p.bhp) and v["WBHP:I1"] == pytest.approx(last_i.bhp)
    assert v["WOPR:P1"] == pytest.approx(last_p.oil_rate) and v["WWPR:P1"] == pytest.approx(last_p.water_rate)
    assert v["WWCT:P1"] == pytest.approx(last_p.water_cut) and v["WWIR:I1"] == pytest.approx(last_i.water_rate)
    assert v["FOPT"] == pytest.approx(last_p.cum_oil) and v["FWPT"] == pytest.approx(last_p.cum_water)
    assert v["FWIT"] == pytest.approx(last_i.cum_water)
    pv = sim.grid.volume * model.porosity
    assert v["FPRP"] == pytest.approx((res.pressure[-1] * pv).sum() / pv.sum())


def test_summary_vectors_before_any_step_has_no_wells_but_has_the_pressure():
    v = summary_vectors(Simulator(_model()))
    assert "WBHP:P1" not in v and "FPRP" in v


# ------------------------------------------------------------------ run_to
def test_run_to_records_the_requested_times():
    record = run_to(_model(), [10.0, 20.0, 30.0], vectors=["WBHP:P1", "WWCT:P1"])
    assert list(record.table.TIME) == [10.0, 20.0, 30.0]
    assert list(record.table.columns) == ["TIME", "WBHP:P1", "WWCT:P1"]
    assert record.state.time == 30.0 and record.ok


def test_run_to_from_a_state_continues_the_run():
    model = _model()
    whole = run_to(model, [20.0, 40.0])
    first = run_to(model, [20.0])
    second = run_to(model, [40.0], state=first.state)
    assert np.array_equal(whole.state.p_o, second.state.p_o)
    assert whole.table.iloc[1]["WBHP:P1"] == second.table.iloc[0]["WBHP:P1"]


def test_run_to_applies_the_schedule():
    schedule = Schedule([ControlEvent(10.0, "P1", target=30.0)])
    record = run_to(_model(), [10.0, 14.0], schedule, vectors=["WOPR:P1", "WWPR:P1"])
    liquid = record.table["WOPR:P1"] + record.table["WWPR:P1"]
    assert np.allclose(liquid, [70.0, 30.0])


def test_run_to_reports_what_it_adjusted_in_the_restart_state():
    model = _model()
    state = run_to(model, [10.0]).state
    assert run_to(model, [20.0], state=state).adjusted == {"n_saturation_clipped": 0}
    state.s_w[:2] = [0.0, 0.99]                          # an update pushed two saturations outside [0.2, 0.8]
    record = run_to(model, [20.0], state=state)
    assert record.adjusted == {"n_saturation_clipped": 2}
    assert run_to(model, [10.0]).adjusted == {}          # a run from time zero has no restart state


def test_run_to_rejects_a_time_that_is_not_after_the_start():
    with pytest.raises(ValueError, match="not after"):
        run_to(_model(), [0.0])


def test_run_record_ok_flags_trouble():
    outside = _model(initial=__import__("kusim").UniformStart(pressure=900.0, sw=0.2))
    assert run_to(outside, [10.0]).ok is False


# ------------------------------------------------------------------ ensembles
def _ensemble(n=3):
    model = _model()
    params = ParameterSet([FieldParameter("kx", "log", ties=("ky", "kz"))])
    rng = np.random.default_rng(3)
    Y = params.vector(model)[:, None] + rng.normal(0, 0.5, size=(model.grid.n_cells, n))
    return params.apply_all(model, Y)


def test_each_member_gives_the_result_of_running_it_alone():
    members = _ensemble()
    records = run_ensemble(members, [20.0, 30.0])
    for model, record in zip(members, records):
        alone = run_to(model, [20.0, 30.0])
        assert np.array_equal(alone.state.p_o, record.state.p_o)
    assert not np.allclose(records[0].state.p_o, records[1].state.p_o)       # the members really differ


def test_members_can_run_in_separate_processes_with_the_same_result():
    members = _ensemble(3)
    serial = run_ensemble(members, [20.0])
    parallel = run_ensemble(members, [20.0], workers=2)
    for a, b in zip(serial, parallel):
        assert np.array_equal(a.state.p_o, b.state.p_o) and np.array_equal(a.state.s_w, b.state.s_w)


def test_one_executor_can_serve_several_calls():
    from concurrent.futures import ProcessPoolExecutor
    members = _ensemble(3)
    serial = run_ensemble(members, [20.0, 40.0])
    with ProcessPoolExecutor(max_workers=2) as pool:
        first = run_ensemble(members, [20.0], executor=pool)
        second = run_ensemble(members, [40.0], states=[r.state for r in first], executor=pool)
    for a, b in zip(serial, second):
        assert np.array_equal(a.state.p_o, b.state.p_o)


def test_members_restart_from_their_own_states():
    members = _ensemble(3)
    first = run_ensemble(members, [20.0])
    second = run_ensemble(members, [40.0], states=[r.state for r in first])
    whole = run_ensemble(members, [40.0])
    for a, b in zip(second, whole):
        assert np.array_equal(a.state.p_o, b.state.p_o)
    with pytest.raises(ValueError, match="one entry per model"):
        run_ensemble(members, [40.0], states=[first[0].state])


def test_data_matrix_has_one_row_per_vector_and_one_column_per_member():
    members = _ensemble(4)
    records = run_ensemble(members, [20.0, 30.0], vectors=["WBHP:P1", "WWCT:P1", "FPRP"])
    D = data_matrix(records, 30.0, ["WBHP:P1", "FPRP"])
    assert D.shape == (2, 4)
    assert D[0, 2] == records[2].table.iloc[1]["WBHP:P1"]
    with pytest.raises(ValueError, match="no data"):
        data_matrix(records, 25.0, ["FPRP"])


# ------------------------------------------------------------------ observations and misfit
def test_observations_at_a_time():
    obs = Observations(pd.DataFrame({"time": [10.0, 10.0, 20.0], "vector": ["WBHP:P1", "WWCT:P1", "WBHP:P1"],
                                     "value": [2500.0, 0.1, 2400.0], "sigma": [5.0, 0.02, 5.0]}))
    assert obs.times == [10.0, 20.0] and obs.vectors == ["WBHP:P1", "WWCT:P1"]
    names, values, sigmas = obs.at(10.0)
    assert names == ["WBHP:P1", "WWCT:P1"] and list(values) == [2500.0, 0.1] and list(sigmas) == [5.0, 0.02]
    with pytest.raises(ValueError, match="sigma"):
        Observations(pd.DataFrame({"time": [1.0], "vector": ["x"], "value": [1.0], "sigma": [0.0]}))
    with pytest.raises(ValueError, match="lacks"):
        Observations(pd.DataFrame({"time": [1.0]}))


def test_synthetic_observations_are_the_truth_plus_noise_and_reproducible():
    truth = run_to(_model(), [10.0, 20.0, 30.0], vectors=["WBHP:P1", "WBHP:I1"]).table
    a = Observations.synthetic(truth, [10.0, 30.0], {"WBHP:P1": 5.0, "WBHP:I1": 5.0}, np.random.default_rng(7))
    b = Observations.synthetic(truth, [10.0, 30.0], {"WBHP:P1": 5.0, "WBHP:I1": 5.0}, np.random.default_rng(7))
    assert a.table.equals(b.table) and len(a.table) == 4
    exact = Observations.synthetic(truth, [10.0], {"WBHP:P1": 1e-9}, np.random.default_rng(7))
    assert exact.table.value[0] == pytest.approx(truth.iloc[0]["WBHP:P1"], abs=1e-6)
    with pytest.raises(ValueError, match="no data"):
        Observations.synthetic(truth, [11.0], {"WBHP:P1": 5.0}, np.random.default_rng(7))


def test_the_objective_is_the_weighted_sum_of_squared_normalised_residuals():
    obs = Observations(pd.DataFrame({"time": [10.0, 20.0], "vector": "WBHP:P1", "value": [2500.0, 2400.0],
                                     "sigma": [10.0, 20.0], "weight": [1.0, 0.5]}))
    simulated = pd.DataFrame({"TIME": [10.0, 20.0], "WBHP:P1": [2520.0, 2360.0]})
    table = misfit(obs, simulated)
    assert objective(table) == pytest.approx(1.0 * 2.0**2 + 0.5 * (-2.0) ** 2)          # (20/10)^2 and 0.5 (-40/20)^2
    assert list(Observations(pd.DataFrame({"time": [1.0], "vector": "x", "value": [1.0], "sigma": [1.0]})).table.weight) == [1.0]
    with pytest.raises(ValueError, match="weight"):
        Observations(pd.DataFrame({"time": [1.0], "vector": "x", "value": [1.0], "sigma": [1.0], "weight": [-1.0]}))


def test_misfit_of_the_truth_is_zero_and_of_a_wrong_model_is_not():
    model = _model()
    times = [10.0, 20.0, 30.0]
    truth = run_to(model, times, vectors=["WBHP:P1", "WWCT:P1"]).table
    obs = Observations(pd.DataFrame({"time": [10.0, 20.0, 30.0], "vector": "WBHP:P1",
                                     "value": truth["WBHP:P1"].to_numpy(), "sigma": 5.0}))
    assert rms(misfit(obs, truth)) == pytest.approx(0.0, abs=1e-9)
    wrong = run_to(model.with_changes(kx=model.kx * 5, ky=model.ky * 5, kz=model.kz * 5), times, vectors=["WBHP:P1"]).table
    table = misfit(obs, wrong)
    assert rms(table) > 1.0 and list(table.columns) == ["time", "vector", "observed", "simulated", "residual", "normalised", "weight"]
    assert np.allclose(table.normalised, table.residual / 5.0)
    with pytest.raises(ValueError, match="no simulated data"):
        misfit(obs, truth.assign(TIME=truth.TIME + 1))
