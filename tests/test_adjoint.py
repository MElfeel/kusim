"""The adjoint (kusim.adjoint, kusim.gradients): the functional copy of the step, its transpose, and the gradients.

Three layers of evidence, each independent of the one before:
  1. the copy of the step reproduces ``Simulator`` exactly;
  2. the transpose (``backward_step``) agrees with finite differences of the copy, step by step (dot-product test);
  3. the gradients of a whole run agree with finite differences of the objective computed by the *simulator*.
"""

import numpy as np
import pandas as pd
import pytest

from kusim import (Equilibrium, Grid, Injector, Producer, PVTTable, ReservoirModel, RockFluidTable, Simulator)
from kusim.adjoint import (_Const, _field_gradients, _new_accumulators, _well_index_derivatives, adjoint_gradient,
                           backward_step, forward_step, forward_trajectory)
from kusim.gradients import HistoryProblem, fd_gradient
from kusim.history import Observations, run_to
from kusim.parameters import FieldParameter, ParameterSet
from kusim.schedule import ControlEvent, ControlParameter, ControlSet, Schedule, advance
from kusim.wells import well_index

ALL_VECTORS = ["WBHP:P1", "WBHP:I1", "WOPR:P1", "WWPR:P1", "WWCT:P1", "WWIR:I1", "FPRP", "FOPT", "FWPT", "FWIT"]


def _model(seed=0, dt=5.0, t_end=200.0, prod=("liquid_rate", 3000.0), inj=("water_rate", 3500.0), min_bhp=None,
           dip=0.0, nx=4, ny=3, nz=3, min_oil_rate=None):
    """A small layered model: wells in every layer, gravity, capillary pressure, water breaks through at about 120 days."""
    rng = np.random.default_rng(seed)
    top = 7500.0 + dip * (np.arange(nx * ny) % nx)                    # dip along x if ``dip`` is not zero
    grid = Grid([250.0] * nx, [250.0] * ny, [25.0] * nz, top_depth=top)
    n = grid.n_cells
    kx = np.exp(rng.normal(5.3, 0.4, n)) * np.repeat([1.0, 3.0, 1.5][:nz], nx * ny)
    rock_fluid = RockFluidTable([.2, .4, .6, .75, .85, 1.], [.85, .35, .1, .02, 0, 0], [0, .05, .2, .4, .6, 1.],
                                [8, 2.5, .8, .3, .15, 0])
    pvt = PVTTable([400., 1000., 5500.], [1.30, 1.24, 1.17], [1.01] * 3, [.9, 1.0, 1.8], [.7] * 3)
    return ReservoirModel(
        grid, kx, 0.8 * kx, 0.1 * kx, rng.uniform(0.18, 0.26, n), 5e-6, pvt, rock_fluid, 50 / 62.4, 64 / 62.4, 0.2, 0.15,
        Equilibrium(4200.0 - 0.433 * (50 / 62.4) * 7580.0, 8000.0), dt, t_end,
        producers=[Producer(grid.column_blocks(1, 1), prod[0], prod[1], radius=0.35, min_bhp=min_bhp, min_oil_rate=min_oil_rate)],
        injectors=[Injector(grid.column_blocks(nx, ny), inj[0], inj[1], radius=0.35)])


def _observations(model, schedule=None, vectors=None, dates=None, seed=7, upstream="potential"):
    """Noisy data from a *different* realisation of the same model (so that the objective is not at its minimum)."""
    truth = _model(seed=1000, dt=model.dt, t_end=model.t_end, prod=(model.producers[0].mode, model.producers[0].target),
                   inj=(model.injectors[0].mode, model.injectors[0].target), min_bhp=model.producers[0].min_bhp,
                   dip=0.0 if model.grid.depth[0] == model.grid.depth[1] else 30.0)
    dates = dates or [float(t) for t in range(40, int(model.t_end) + 1, 40)]
    vectors = vectors or ALL_VECTORS
    table = run_to(truth, dates, schedule, vectors, upstream=upstream).table
    sigma = {v: (10.0 if v.startswith(("WBHP", "FPRP")) else 0.02 if v == "WWCT:P1" else 20.0 if v.startswith("W") else 5000.0)
             for v in vectors}
    return Observations.synthetic(table, dates, sigma, np.random.default_rng(seed))


def _params():
    return ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=1e4, ties=("ky", "kz")),
                         FieldParameter("porosity", "logit", lower=0.05, upper=0.4)])


# ====================================================================== 1. the copy reproduces the simulator
SCENARIOS = {
    "base": dict(),
    "dipping layers": dict(dip=30.0),
    "bhp wells": dict(prod=("bhp", 2500.0), inj=("bhp", 5200.0)),
    "oil_rate producer": dict(prod=("oil_rate", 1500.0), t_end=120.0),
    "producer reaches min_bhp": dict(prod=("liquid_rate", 9000.0), min_bhp=3300.0),
    "min_oil_rate stop": dict(prod=("liquid_rate", 3000.0), min_oil_rate=2900.0),
}


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("upstream", ["potential", "pressure"])
def test_the_functional_copy_reproduces_the_simulator_exactly(name, upstream):
    model = _model(**SCENARIOS[name])
    schedule = Schedule([ControlEvent(50.0, "I1", target=3000.0)]) if name == "base" else None
    sim = Simulator(model, save_every=1, upstream=upstream)
    advance(sim, model.t_end, schedule)
    res = sim.results()
    traj = forward_trajectory(model, model.t_end, schedule, upstream=upstream)
    assert len(traj.p) == len(res.pressure) and len(traj.times) == len(traj.p) - 1
    assert all(np.array_equal(a, b) for a, b in zip(traj.p, res.pressure))
    assert all(np.array_equal(a, b) for a, b in zip(traj.s, res.sw))
    wells = res.wells
    for i in range(len(traj.times)):
        assert traj.outputs[i]["P1"]["bhp"] == wells[wells.well == "P1"].bhp.iloc[i]
        assert traj.outputs[i]["I1"]["qw"] == wells[wells.well == "I1"].water_rate.iloc[i]
    assert traj.stop_reason == sim.stop_reason
    assert (name == "min_oil_rate stop") == (traj.stop_reason is not None)             # the stop is really exercised


def test_the_copy_follows_the_min_bhp_switch_and_reports_it():
    traj = forward_trajectory(_model(**SCENARIOS["producer reaches min_bhp"]), 200.0)
    assert traj.diagnostics["mode_switches"] == 1
    modes = [c["P1"][0] for c in traj.controls]
    assert modes[0] == "liquid_rate" and modes[-1] == "bhp"
    assert traj.scheduled[0]["P1"] is True and traj.scheduled[-1]["P1"] is False       # the limit is not a schedule target


def test_the_copy_can_restart_from_a_state():
    model = _model()
    first = run_to(model, [100.0])
    traj = forward_trajectory(model, 200.0, state=first.state)
    sim = Simulator(model, save_every=1)
    sim.set_state(first.state)
    advance(sim, 200.0)
    assert traj.times[0] == 100.0 and np.array_equal(traj.p[-1], sim.p_o)


# ====================================================================== 2. the transpose, step by step
COMBOS = [("liquid_rate", 6000.0, "water_rate", 7000.0), ("oil_rate", 3500.0, "bhp", 5200.0),
          ("bhp", 2800.0, "water_rate", 6500.0), ("liquid_rate", 5000.0, "bhp", 5000.0)]


def _step_value(model, p_old, s_old, controls, weights):
    c = _Const(Simulator(model, save_every=10**9))
    p_new, s_new, outs, _ = forward_step(c, p_old, s_old, controls)
    value = weights["p"] @ p_new + weights["s"] @ s_new
    for name, o in outs.items():
        value += sum(weights["w"][name][key] * o[key] for key in ("bhp", "qo", "qw"))
    return value


def _richardson(f, h):
    """Central difference with one Richardson step: error of order h**4 (the round-off stays small for a moderate h)."""
    d = lambda step: (f(+step) - f(-step)) / (2 * step)
    return (4 * d(h / 2) - d(h)) / 3


@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: f"{c[0]}+{c[2]}")
def test_the_backward_step_is_the_transpose_of_the_forward_step(combo):
    pm, pt, im, it = combo
    rng = np.random.default_rng(5)
    model = _model().with_changes({"producers[0].mode": pm, "producers[0].target": pt,
                                   "injectors[0].mode": im, "injectors[0].target": it})
    n = model.grid.n_cells
    sim = Simulator(model, save_every=10**9)
    advance(sim, 25.0)                                                        # a state from the middle of a run
    p_old, s_old = sim.p_o.copy(), sim.s_w.copy()
    controls = {"P1": (pm, pt), "I1": (im, it)}
    weights = {"p": rng.normal(size=n) * 1e-2, "s": rng.normal(size=n),
               "w": {"P1": dict(bhp=0.01, qo=0.02, qw=-0.01), "I1": dict(bhp=-0.01, qo=0.0, qw=0.015)}}
    c = _Const(sim)
    _, _, _, k = forward_step(c, p_old, s_old, controls)
    acc = _new_accumulators(c)
    p_bar, s_bar = backward_step(c, k, weights["p"], weights["s"], {w: dict(v) for w, v in weights["w"].items()}, acc)
    fields = _field_gradients(model, c, acc)

    dp, ds = rng.normal(size=n), rng.normal(size=n) * 0.1
    nodes = np.array([0.2, 0.4, 0.6, 0.75, 0.85, 1.0])               # the saturation nodes of the tables, and the clip limits
    ds[np.abs(s_old[:, None] - nodes[None, :]).min(axis=1) < 1e-3] = 0.0     # a block ON a node has only a one-sided derivative
    value = lambda m=model, p=p_old, s=s_old, ctl=controls: _step_value(m, p, s, ctl, weights)
    checks = [("p_old", p_bar @ dp, _richardson(lambda e: value(p=p_old + e * dp), 2e-3)),
              ("s_old", s_bar @ ds, _richardson(lambda e: value(s=s_old + e * ds), 2e-4))]
    for field_name, scale, h in (("kx", 5.0, 1e-2), ("ky", 5.0, 1e-2), ("kz", 5.0, 1e-2), ("porosity", 0.01, 1e-2)):
        d = rng.normal(size=n) * scale
        checks.append((field_name, fields[field_name] @ d,
                       _richardson(lambda e: value(m=model.with_changes({field_name: getattr(model, field_name) + e * d})), h)))
    for well in ("P1", "I1"):
        def shifted(e, well=well):
            ctl = dict(controls)
            ctl[well] = (controls[well][0], controls[well][1] + e)
            return value(ctl=ctl)
        checks.append((f"target {well}", acc.ctrl_bar[well], _richardson(shifted, 1.0)))
    for label, analytic, fd in checks:
        assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-12), label


def test_the_well_index_derivatives_match_finite_differences():
    for kx, ky in [(200.0, 160.0), (150.0, 450.0), (80.0, 80.0)]:
        pid, dkx, dky = _well_index_derivatives(kx, ky, 250.0, 300.0, 25.0, 0.35, 1.5)
        assert pid == well_index(kx, ky, 250.0, 300.0, 25.0, 0.35, 1.5)
        h = 1e-4
        fd_x = (well_index(kx + h, ky, 250.0, 300.0, 25.0, 0.35, 1.5) - well_index(kx - h, ky, 250.0, 300.0, 25.0, 0.35, 1.5)) / (2 * h)
        fd_y = (well_index(kx, ky + h, 250.0, 300.0, 25.0, 0.35, 1.5) - well_index(kx, ky - h, 250.0, 300.0, 25.0, 0.35, 1.5)) / (2 * h)
        assert dkx == pytest.approx(fd_x, rel=1e-7) and dky == pytest.approx(fd_y, rel=1e-7)


# ====================================================================== 3. the gradients of whole runs
def _check(problem, x, n_params=12, tol=1e-5, min_stable=0.7):
    """Adjoint against finite differences (through the simulator) for all controls and some parameters.

    A finite-difference estimate counts only when two neighbouring step sizes agree (``HistoryProblem.check``); the objective
    has kinks, so a few components may have no stable estimate, but most must, and the adjoint must match every stable one."""
    indices = list(np.linspace(0, problem.n_params - 1, n_params, dtype=int)) if problem.n_params else []
    indices += list(range(problem.n_params, problem.n_params + problem.n_controls))
    table = problem.check(x, indices=indices)
    assert table.stable.mean() >= min_stable, table.to_string()
    assert table.relative.max() < tol, table.to_string()
    return table


def test_objective_by_adjoint_copy_and_by_simulator_agree_exactly():
    model = _model()
    problem = HistoryProblem(model, _observations(model), params=_params())
    x = problem.vector0()
    value, _, result = problem.value_and_gradient(x)
    assert value == result.value
    assert value == pytest.approx(problem.value(x), rel=1e-12)             # a different order of summation: equal to rounding


def test_gradient_of_all_vectors_with_respect_to_permeability_and_porosity():
    model = _model()
    problem = HistoryProblem(model, _observations(model), params=_params())
    _check(problem, problem.vector0())


def test_gradient_with_respect_to_well_controls_in_every_mode_pair():
    for pm, pt, im, it in COMBOS:
        model = _model(prod=(pm, pt), inj=(im, it), t_end=120.0)
        controls = ControlSet([ControlParameter("P1", [0.0, 40.0, 80.0], mode=pm, upper=20000.0),
                               ControlParameter("I1", [0.0, 60.0], mode=im, upper=20000.0)])
        initial = {"P1": [pt, 0.9 * pt, 0.8 * pt], "I1": [it, 0.9 * it]}
        schedule = controls.schedule(controls.vector(initial))
        obs = _observations(model, schedule, dates=[30.0, 60.0, 90.0, 120.0])
        problem = HistoryProblem(model, obs, params=_params(), controls=controls, initial_controls=initial)
        _check(problem, problem.vector0(), n_params=6)


def test_gradient_with_a_schedule_a_dipping_model_and_the_pressure_rule():
    for upstream in ("potential", "pressure"):
        model = _model(dip=30.0)
        history = Schedule([ControlEvent(60.0, "P1", target=2500.0), ControlEvent(100.0, "I1", target=3000.0)])
        controls = ControlSet([ControlParameter("P1", [0.0, 140.0], mode="liquid_rate")])
        problem = HistoryProblem(model, _observations(model, history, upstream=upstream), params=_params(), controls=controls,
                                 base_schedule=history, initial_controls={"P1": [3000.0, 2800.0]}, upstream=upstream)
        # a control with the history's event at day 60 replaced: the controlled well takes its schedule from the ControlSet
        _check(problem, problem.vector0(), n_params=8)


def test_gradient_through_a_producer_that_switches_to_its_bhp_limit():
    model = _model(prod=("liquid_rate", 9000.0), min_bhp=3300.0)
    controls = ControlSet([ControlParameter("P1", [0.0, 100.0], mode="liquid_rate"), ControlParameter("I1", [0.0], mode="water_rate")])
    problem = HistoryProblem(model, _observations(model), params=_params(), controls=controls,
                             initial_controls={"P1": [9000.0, 9000.0], "I1": [3500.0]})
    x = problem.vector0()
    _, g, result = problem.value_and_gradient(x)
    assert result.diagnostics["mode_switches"] >= 1                          # the event at day 100 sets liquid_rate again: a second switch
    switched = np.array([not s["P1"] for s in result.trajectory.scheduled])
    assert switched.any() and np.all(result.controls["P1"][switched] == 0.0)        # after the switch the rate is not the control
    _check(problem, x, n_params=8)


def test_gradient_from_a_restart_state():
    model = _model(t_end=300.0)
    state = run_to(model, [100.0]).state
    dates = [140.0, 200.0, 260.0, 300.0]
    controls = ControlSet([ControlParameter("P1", [100.0, 200.0], mode="liquid_rate")])
    initial = {"P1": [3000.0, 3000.0]}
    obs = _observations(model, controls.schedule(controls.vector(initial)), dates=dates)
    problem = HistoryProblem(model, obs, params=_params(), controls=controls, initial_controls=initial, state=state)
    _check(problem, problem.vector0(), n_params=6)


def test_weights_scale_the_objective_and_zero_weight_removes_a_datum():
    model = _model()
    obs = _observations(model, vectors=["WBHP:P1", "WWCT:P1"])
    problem = HistoryProblem(model, obs, params=_params())
    x = problem.vector0()
    J1, g1, _ = problem.value_and_gradient(x)
    doubled = Observations(obs.table.assign(weight=2.0))
    J2, g2, _ = HistoryProblem(model, doubled, params=_params()).value_and_gradient(x)
    assert J2 == pytest.approx(2 * J1) and np.allclose(g2, 2 * g1)
    only_bhp = Observations(obs.table.assign(weight=np.where(obs.table.vector == "WBHP:P1", 1.0, 0.0)))
    J3, g3, _ = HistoryProblem(model, only_bhp, params=_params()).value_and_gradient(x)
    J4, g4, _ = HistoryProblem(model, Observations(obs.table[obs.table.vector == "WBHP:P1"]), params=_params()).value_and_gradient(x)
    assert J3 == pytest.approx(J4) and np.allclose(g3, g4)


def test_a_missing_observation_is_skipped():
    model = _model()
    obs = _observations(model, vectors=["WBHP:P1"])
    table = obs.table.copy()
    table.loc[0, "value"] = np.nan
    result = adjoint_gradient(model, Observations(table))
    assert np.isfinite(result.value)
    assert result.value == pytest.approx(adjoint_gradient(model, Observations(table.iloc[1:])).value)


def test_the_adjoint_is_much_cheaper_than_finite_differences():
    import time
    model = _model()
    problem = HistoryProblem(model, _observations(model), params=_params())
    x = problem.vector0()
    start = time.perf_counter()
    problem.value_and_gradient(x)
    adjoint_time = time.perf_counter() - start
    start = time.perf_counter()
    problem.value(x)
    one_run = time.perf_counter() - start
    assert adjoint_time < 12 * one_run                       # the cost of a few runs, for 2 x 36 parameters


def test_diagnostics_count_the_decisions_that_make_the_objective_non_smooth():
    model = _model(prod=("liquid_rate", 9000.0), min_bhp=3300.0)
    result = adjoint_gradient(model, _observations(model, vectors=["WBHP:P1"]))
    d = result.diagnostics
    assert set(d) >= {"steps", "upstream_flips", "mode_switches", "saturation_clipped", "stop_reason"}
    assert d["steps"] == 40 and d["mode_switches"] == 1 and d["upstream_flips"] > 0


def test_the_initial_state_gradients_have_the_shape_of_the_state():
    model = _model()
    result = adjoint_gradient(model, _observations(model, vectors=["WBHP:P1"]))
    assert result.initial_pressure.shape == result.initial_saturation.shape == (model.grid.n_cells,)
    assert np.any(result.initial_pressure != 0.0)


# ====================================================================== errors and the helpers
def test_unsupported_vectors_and_bad_observation_times_are_rejected():
    model = _model()
    bad_vector = Observations(pd.DataFrame({"time": [40.0], "vector": ["WGOR:P1"], "value": [1.0], "sigma": [1.0]}))
    with pytest.raises(ValueError, match="does not support"):
        adjoint_gradient(model, bad_vector)
    off_step = Observations(pd.DataFrame({"time": [42.0], "vector": ["WBHP:P1"], "value": [1.0], "sigma": [1.0]}))
    with pytest.raises(ValueError, match="whole number"):
        adjoint_gradient(model, off_step)
    at_start = Observations(pd.DataFrame({"time": [0.0], "vector": ["WBHP:P1"], "value": [1.0], "sigma": [1.0]}))
    with pytest.raises(ValueError):
        adjoint_gradient(model, at_start)


def test_an_observation_after_an_early_stop_is_rejected():
    model = _model(**SCENARIOS["min_oil_rate stop"])
    stopped_at = forward_trajectory(model, 200.0).times[-1]
    assert stopped_at < 190.0
    obs = Observations(pd.DataFrame({"time": [190.0], "vector": ["WBHP:P1"], "value": [1.0], "sigma": [1.0]}))
    with pytest.raises(ValueError, match="stopped"):
        adjoint_gradient(model, obs)


def test_control_set_builds_schedules_and_pulls_gradients_back():
    controls = ControlSet([ControlParameter("P1", [0.0, 100.0, 200.0], mode="liquid_rate", upper=5000.0),
                           ControlParameter("I1", [0.0], lower=None)])
    assert controls.size == 4
    base = Schedule([ControlEvent(50.0, "P1", target=1.0), ControlEvent(60.0, "X1", target=2.0)])
    vector = controls.vector({"P1": [1000.0, 9000.0, 3000.0], "I1": [-5.0]})
    schedule = controls.schedule(vector, base)
    targets = {(e.time, e.well): e.target for e in schedule.events}
    assert targets == {(60.0, "X1"): 2.0, (0.0, "P1"): 1000.0, (100.0, "P1"): 5000.0, (200.0, "P1"): 3000.0, (0.0, "I1"): -5.0}
    assert all(e.mode == "liquid_rate" for e in schedule.events if e.well == "P1")      # the base event of P1 was replaced
    step_times = np.arange(0.0, 300.0, 50.0)                                            # steps starting at 0, 50, ..., 250
    g = controls.pullback(vector, {"P1": np.arange(6, dtype=float) + 1, "I1": np.ones(6)}, step_times)
    assert list(g) == [1 + 2, 0.0, 5 + 6, 6.0]               # periods [0, 100): steps 0, 50; [100, 200): clipped (9000 > 5000) -> 0
    lower, upper = controls.bounds()
    assert list(lower) == [0.0, 0.0, 0.0, -np.inf] and list(upper) == [5000.0, 5000.0, 5000.0, np.inf]
    with pytest.raises(ValueError, match="increase"):
        ControlParameter("P1", [0.0, 0.0])
    with pytest.raises(ValueError, match="one ControlParameter per well"):
        ControlSet([ControlParameter("P1", [0.0]), ControlParameter("P1", [10.0])])
    with pytest.raises(ValueError, match="targets are needed"):
        controls.vector({"P1": [1.0], "I1": [1.0]})
    with pytest.raises(ValueError, match="values"):
        controls.schedule(np.zeros(3))


def test_fd_gradient_helper():
    f = lambda x: float(x[0] ** 2 + 3.0 * x[1] + np.sin(x[2]))
    x = np.array([2.0, -1.0, 0.5])
    exact = np.array([4.0, 3.0, np.cos(0.5)])
    assert np.allclose(fd_gradient(f, x, h=1e-5), exact, rtol=1e-8)
    assert np.allclose(fd_gradient(f, x, h=1e-6, central=False), exact, rtol=1e-4)
    partial = fd_gradient(f, x, indices=[1])
    assert partial[1] == pytest.approx(3.0) and np.isnan(partial[0]) and np.isnan(partial[2])

    class Pool:                                                       # anything with ``map`` works as an executor
        def map(self, fn, points):
            return [fn(p) for p in points]
    assert np.allclose(fd_gradient(f, x, executor=Pool()), exact, rtol=1e-6)


def test_history_problem_validates_its_arguments():
    model = _model()
    obs = _observations(model, vectors=["WBHP:P1"])
    with pytest.raises(ValueError, match="nothing to vary"):
        HistoryProblem(model, obs)
    with pytest.raises(ValueError, match="initial_controls"):
        HistoryProblem(model, obs, controls=ControlSet([ControlParameter("P1", [0.0])]))
    problem = HistoryProblem(model, obs, params=_params())
    with pytest.raises(ValueError, match="values"):
        problem.value(np.zeros(3))
    lower, upper = problem.bounds()
    assert lower.shape == upper.shape == (problem.n_params,)


# ====================================================================== linear (economic) terms and a prior
def test_gradient_of_an_economic_objective_with_respect_to_the_controls():
    from kusim.history import linear_terms
    model = _model(t_end=200.0)
    controls = ControlSet([ControlParameter("I1", [0.0, 50.0, 100.0, 150.0], mode="water_rate", upper=9000.0),
                           ControlParameter("P1", [0.0, 100.0], mode="liquid_rate", upper=9000.0)])
    initial = {"I1": [3500.0, 3000.0, 4000.0, 2500.0], "P1": [3000.0, 3200.0]}
    profit = linear_terms(200.0, {"FOPT": -60.0, "FWPT": 5.0, "FWIT": 1.0})        # J = minus the profit
    problem = HistoryProblem(model, controls=controls, initial_controls=initial, linear_terms=profit)
    x = problem.vector0()
    value, g, result = problem.value_and_gradient(x)
    assert value == pytest.approx(problem.value(x), rel=1e-12)
    assert value == pytest.approx(-60.0 * result.simulated[(200.0, "FOPT")] + 5.0 * result.simulated[(200.0, "FWPT")]
                                  + result.simulated[(200.0, "FWIT")])
    table = _check(problem, x)
    assert table.stable.all()
    assert g[-2] < 0 and g[0] != 0                                              # more production early earns money


def test_observations_and_linear_terms_and_a_prior_can_be_combined():
    from kusim.history import linear_terms
    model = _model()
    prior_mean = _params().vector(model)
    problem = HistoryProblem(model, _observations(model, vectors=["WBHP:P1", "WWCT:P1"]), params=_params(),
                             linear_terms=linear_terms(200.0, {"FOPT": -0.001}), prior=(prior_mean, 0.5))
    x = problem.vector0() + 0.05
    value, g, _ = problem.value_and_gradient(x)
    assert value == pytest.approx(problem.value(x), rel=1e-12)
    plain = HistoryProblem(model, _observations(model, vectors=["WBHP:P1", "WWCT:P1"]), params=_params(),
                           linear_terms=linear_terms(200.0, {"FOPT": -0.001}))
    n = problem.n_params
    assert value - plain.value(x) == pytest.approx(np.sum(((x - prior_mean) / 0.5) ** 2))      # the prior term
    assert np.allclose(g - plain.value_and_gradient(x)[1], 2 * (x - prior_mean) / 0.25)
    _check(problem, x)


def test_a_prior_pulls_towards_its_mean():
    model = _model()
    params = _params()
    prior_mean = params.vector(model)
    problem = HistoryProblem(model, _observations(model, vectors=["WBHP:P1"]), params=params, prior=(prior_mean, 0.1))
    far = prior_mean + 1.0
    _, g, _ = problem.value_and_gradient(far)
    data_only = HistoryProblem(model, _observations(model, vectors=["WBHP:P1"]), params=params).value_and_gradient(far)[1]
    assert np.allclose(g - data_only, 2.0 * 1.0 / 0.1**2)                        # the pull of the prior at one unit away


def test_objective_inputs_are_validated():
    from kusim.history import linear_terms
    model = _model()
    with pytest.raises(ValueError, match="no objective"):
        adjoint_gradient(model)
    with pytest.raises(ValueError, match="no objective"):
        HistoryProblem(model, params=_params())
    with pytest.raises(ValueError, match="prior is on the parameters"):
        HistoryProblem(model, _observations(model), controls=ControlSet([ControlParameter("P1", [0.0])]),
                       initial_controls={"P1": [3000.0]}, prior=(0.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        HistoryProblem(model, _observations(model), params=_params(), prior=(0.0, 0.0))
    with pytest.raises(ValueError, match="does not support"):
        adjoint_gradient(model, linear_terms=linear_terms(200.0, {"WGOR:P1": 1.0}))


# ====================================================================== two pitfalls, kept as tests so that they stay documented
def test_backflow_is_flagged_by_the_simulator_and_the_adjoint():
    ok = _model(prod=("bhp", 2500.0), inj=("bhp", 5200.0))
    sim = Simulator(ok)
    advance(sim, 100.0)
    assert sim.diagnostics()["n_backflow_well_steps"] == 0
    assert adjoint_gradient(ok, _observations(ok, vectors=["WBHP:P1"])).diagnostics["backflow_well_steps"] == 0
    # a producer whose bottom-hole pressure is above the pressure of its blocks flows backwards (a producer that injects)
    backwards = _model(prod=("bhp", 9000.0), inj=("bhp", 5200.0))
    sim = Simulator(backwards)
    advance(sim, 100.0)
    assert sim.diagnostics()["n_backflow_well_steps"] > 0
    assert adjoint_gradient(backwards, _observations(backwards, vectors=["WBHP:P1"])).diagnostics["backflow_well_steps"] > 0


def test_the_objective_jumps_when_a_producer_reaches_its_bhp_limit():
    """The limit is applied after the step as a switch, so the objective is discontinuous in the rate: the adjoint is the
    derivative on one side of the jump (this is why an optimiser must not be allowed to sit on an active limit)."""
    model = _model(prod=("liquid_rate", 3000.0), min_bhp=3300.0)
    profit = pd.DataFrame([dict(time=200.0, vector="FOPT", weight=-1.0)])
    problem_at = lambda target: HistoryProblem(
        model.with_changes({"producers[0].target": target}),
        controls=ControlSet([ControlParameter("P1", [0.0], mode="liquid_rate")]), initial_controls={"P1": [target]}, linear_terms=profit)
    switches = lambda target: adjoint_gradient(model.with_changes({"producers[0].target": target}), linear_terms=profit).diagnostics["mode_switches"]
    low, high = 3000.0, 9000.0
    assert switches(low) == 0 and switches(high) >= 1
    for _ in range(60):                                                    # bisection for the edge of the jump
        mid = 0.5 * (low + high)
        low, high = (mid, high) if switches(mid) == 0 else (low, mid)
    assert high - low < 1e-6
    j_low, g_low, _ = problem_at(low).value_and_gradient(np.array([low]))
    j_high, g_high, _ = problem_at(high).value_and_gradient(np.array([high]))
    smooth_change = abs(g_low[0]) * (high - low)
    assert abs(j_high - j_low) > 1e3 * max(smooth_change, 1e-6)             # a jump, not a slope
    assert abs(j_high - j_low) > 1.0                                        # more than a barrel, for controls 1e-6 STB/day apart
