"""Restart from a saved state, public well controls, schedules and run diagnostics."""

import numpy as np
import pytest

from kusim import Injector, Producer, Simulator, UniformStart
from kusim.schedule import ControlEvent, Schedule, advance

from conftest import make_model


def _model(**kw):
    defaults = dict(nx=6, ny=1, nz=1, dt=2.0, t_end=60.0, pc=True,
                    injectors=[Injector(1, "water_rate", 60.0)], producers=[Producer(6, "liquid_rate", 70.0, min_bhp=1000.0)])
    defaults.update(kw)
    return make_model(**defaults)


# ------------------------------------------------------------------ restart
def test_a_restart_gives_exactly_the_same_result_as_one_run():
    model = _model()
    whole = Simulator(model)
    advance(whole, 60.0)

    first = Simulator(model)
    advance(first, 30.0)
    state = first.get_state()
    second = Simulator(model)                       # a different simulator object, built from the same model
    second.set_state(state)
    advance(second, 60.0)

    assert np.array_equal(whole.p_o, second.p_o) and np.array_equal(whole.s_w, second.s_w)
    assert second.time == 60.0 and second.step_number == whole.step_number
    assert whole.get_state().cum["P1"] == second.get_state().cum["P1"]          # cumulative volumes continue too


def test_the_state_is_a_snapshot_and_not_a_view():
    sim = Simulator(_model())
    advance(sim, 10.0)
    state = sim.get_state()
    p_then = state.p_o.copy()
    advance(sim, 30.0)
    assert np.array_equal(state.p_o, p_then) and state.time == 10.0


def test_a_state_can_continue_in_a_model_with_other_permeabilities():
    model = _model()
    sim = Simulator(model)
    advance(sim, 20.0)
    changed = model.with_changes(kx=model.kx * 2, ky=model.ky * 2, kz=model.kz * 2)
    other = Simulator(changed)
    other.set_state(sim.get_state())
    advance(other, 60.0)
    advance(sim, 60.0)
    assert not np.allclose(other.p_o, sim.p_o)                    # the edit made a difference...
    assert other.time == 60.0 and np.all(np.isfinite(other.p_o))  # ...and the run is fine


def test_the_state_carries_a_producer_that_switched_to_bhp_control():
    model = _model(producers=[Producer(6, "liquid_rate", 400.0, min_bhp=2950.0)])      # hits the limit quickly
    sim = Simulator(model)
    advance(sim, 30.0)
    assert sim.get_state().well_controls["P1"][0] == "bhp"
    other = Simulator(model)
    other.set_state(sim.get_state())
    assert other.get_state().well_controls["P1"] == sim.get_state().well_controls["P1"]


def test_set_state_clips_saturation_and_reports_it():
    sim = Simulator(_model())
    state = sim.get_state()
    state.s_w = np.array([0.0, 0.5, 0.95, 0.3, 0.2, 0.8])         # 0.0 and 0.95 are outside [0.2, 0.8]
    adjusted = sim.set_state(state)
    assert adjusted == {"n_saturation_clipped": 2}
    assert sim.s_w.min() >= 0.2 and sim.s_w.max() <= 0.8
    with pytest.raises(ValueError, match="outside"):
        sim.set_state(state, clip=False)


def test_set_state_rejects_inconsistent_states():
    sim = Simulator(_model())
    good = sim.get_state()
    bad_shape = good.copy(); bad_shape.p_o = np.zeros(3)
    bad_nan = good.copy(); bad_nan.p_o[0] = np.nan
    bad_time = good.copy(); bad_time.time = 3.0                    # dt = 2
    bad_wells = good.copy(); bad_wells.well_controls = {"P1": ("bhp", 2000.0)}
    for state, message in [(bad_shape, "needs"), (bad_nan, "NaN"), (bad_time, "multiple"), (bad_wells, "wells")]:
        with pytest.raises(ValueError, match=message):
            sim.set_state(state)


def test_a_restart_begins_a_new_record_of_the_run():
    sim = Simulator(_model())
    advance(sim, 20.0)
    other = Simulator(_model())
    other.set_state(sim.get_state())
    assert other.well_data() == {}
    advance(other, 30.0)
    res = other.results()
    assert res.times[0] == 20.0 and res.times[-1] == 30.0


# ------------------------------------------------------------------ well controls
def test_set_well_control_changes_the_rate_from_the_next_step():
    sim = Simulator(_model())
    advance(sim, 10.0)
    assert np.isclose(sim.well_data()["P1"]["oil_rate"] + sim.well_data()["P1"]["water_rate"], 70.0)
    sim.set_well_control("P1", target=30.0)
    sim.step()
    assert np.isclose(sim.well_data()["P1"]["oil_rate"] + sim.well_data()["P1"]["water_rate"], 30.0)


def test_set_well_control_checks_its_arguments():
    sim = Simulator(_model())
    with pytest.raises(KeyError, match="P9"):
        sim.set_well_control("P9", target=1.0)
    with pytest.raises(ValueError, match="mode"):
        sim.set_well_control("I1", mode="liquid_rate")                 # an injector cannot take a producer mode
    with pytest.raises(ValueError, match="mode"):
        sim.set_well_control("P1", mode="water_rate")
    with pytest.raises(ValueError, match="target"):
        sim.set_well_control("P1", target=-5.0)
    with pytest.raises(ValueError, match="target"):
        sim.set_well_control("P1", target=float("nan"))


def test_a_rate_mode_can_be_restored_after_the_bhp_limit():
    sim = Simulator(_model(producers=[Producer(6, "liquid_rate", 400.0, min_bhp=2950.0)]))
    advance(sim, 30.0)
    assert sim.get_state().well_controls["P1"][0] == "bhp"
    sim.set_well_control("P1", mode="liquid_rate", target=20.0)
    assert sim.get_state().well_controls["P1"] == ("liquid_rate", 20.0)


# ------------------------------------------------------------------ schedule
def _liquid(sim):
    d = sim.well_data()["P1"]
    return d["oil_rate"] + d["water_rate"]


def test_events_are_applied_at_their_time():
    sim = Simulator(_model(t_end=40.0))
    schedule = Schedule([ControlEvent(10.0, "P1", target=30.0), ControlEvent(20.0, "P1", target=50.0)])
    seen = {}
    for t in (8.0, 10.0, 12.0, 20.0, 22.0):
        advance(sim, t, schedule)
        seen[t] = _liquid(sim)
    assert np.isclose(seen[8.0], 70.0)                       # before the first event: the rate of the model
    assert np.isclose(seen[10.0], 70.0)                      # the step that ENDS at day 10 started before the event
    assert np.isclose(seen[12.0], 30.0)                      # the step from 10 to 12 uses the new rate
    assert np.isclose(seen[20.0], 30.0)
    assert np.isclose(seen[22.0], 50.0)


def test_a_schedule_can_change_the_mode_and_close_a_well():
    sim = Simulator(_model(t_end=40.0))
    schedule = Schedule([ControlEvent(10.0, "I1", target=0.0), ControlEvent(20.0, "P1", "bhp", 2800.0)])
    advance(sim, 14.0, schedule)
    assert sim.well_data()["I1"]["water_rate"] == 0.0
    advance(sim, 24.0, schedule)
    assert sim.well_data()["P1"]["mode"] == "bhp" and sim.well_data()["P1"]["bhp"] == 2800.0


def test_a_restart_in_the_middle_of_a_schedule_gives_the_same_result():
    model = _model(t_end=40.0)
    schedule = Schedule([ControlEvent(10.0, "P1", target=30.0), ControlEvent(20.0, "P1", target=50.0), ControlEvent(30.0, "I1", target=20.0)])
    whole = Simulator(model)
    advance(whole, 40.0, schedule)
    part = Simulator(model)
    advance(part, 16.0, schedule)
    rest = Simulator(model)
    rest.set_state(part.get_state())
    advance(rest, 40.0, schedule)                       # the event at day 10 is in the past and is not applied again
    assert np.array_equal(whole.p_o, rest.p_o) and np.array_equal(whole.s_w, rest.s_w)


def test_advance_checks_times():
    sim = Simulator(_model(t_end=40.0))
    with pytest.raises(ValueError, match="whole number"):
        advance(sim, 5.0)                                # dt = 2
    with pytest.raises(ValueError, match="beyond"):
        advance(sim, 100.0)
    with pytest.raises(ValueError, match="multiple"):
        advance(sim, 10.0, Schedule([ControlEvent(5.0, "P1", target=1.0)]))
    advance(sim, 10.0)
    with pytest.raises(ValueError, match="whole number"):
        advance(sim, 4.0)                                # backwards


def test_schedule_from_a_table_and_validation():
    import pandas as pd
    table = pd.DataFrame({"time": [0.0, 10.0], "well": ["P1", "I1"], "mode": ["liquid_rate", None], "target": [60.0, 40.0]})
    schedule = Schedule.from_table(table)
    assert len(schedule) == 2 and schedule.events[1].mode is None and schedule.times() == [0.0, 10.0]
    with pytest.raises(ValueError, match="changes nothing"):
        ControlEvent(5.0, "P1")
    with pytest.raises(ValueError, match="negative"):
        ControlEvent(-1.0, "P1", target=1.0)
    assert len(schedule.with_event(ControlEvent(5.0, "P1", target=1.0))) == 3


# ------------------------------------------------------------------ diagnostics
def test_diagnostics_flag_a_run_that_leaves_the_pvt_table():
    inside = Simulator(_model())
    advance(inside, 20.0)
    assert inside.diagnostics()["outside_pvt_range"] is False
    outside = Simulator(_model(initial=UniformStart(pressure=900.0, sw=0.2)))          # the table starts at 1000
    assert outside.diagnostics()["outside_pvt_range"] is True


def test_diagnostics_count_saturation_clipping():
    quiet = Simulator(_model())
    advance(quiet, 40.0)
    assert quiet.diagnostics()["n_saturation_clipped"] == 0
    # a huge time step with strong wells makes the explicit saturation update overshoot
    rough = Simulator(_model(dt=20.0, t_end=200.0, injectors=[Injector(1, "water_rate", 5000.0)],
                             producers=[Producer(6, "liquid_rate", 5000.0)]))
    advance(rough, 200.0)
    assert rough.diagnostics()["n_saturation_clipped"] > 0
