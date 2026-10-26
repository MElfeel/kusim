"""Physics checks on the IMPES simulator."""

import copy

import numpy as np
import pytest

from kusim import Equilibrium, Injector, Producer, Simulator, UniformStart
from kusim.constants import BBL_TO_FT3

from conftest import make_model


# ------------------------------------------------------------ equilibrium
def test_equilibrium_start_is_stationary(legacy_model):
    """With no wells nothing should move: the initial state is in capillary-gravity equilibrium."""
    m = copy.deepcopy(legacy_model)
    m.producers, m.injectors = [], []
    m.dt, m.t_end = 5.0, 50.0
    sim = Simulator(m)
    p0, s0 = sim.p_o.copy(), sim.s_w.copy()
    sim.run()
    assert np.max(np.abs(sim.p_o - p0)) < 1e-6
    assert np.max(np.abs(sim.s_w - s0)) < 1e-8


def test_legacy_initial_state_matches_shipped_baseline_first_step(legacy_model):
    """First-step well rates for INPUT.DAT equal the values in the old QWP.OUT / QLP.OUT.

    The first-step water/oil ratio depends only on the initial saturation and the
    fluid tables, so it is independent of the later numerical differences.
    """
    sim = Simulator(legacy_model)
    sim.step()
    row = sim.results().wells.iloc[0]
    assert row.oil_rate == pytest.approx(5000.0)
    # rel=1e-7, not exact: the FORTRAN used single-precision constants such as 0.433
    assert row.water_rate == pytest.approx(253.175168898994, rel=1e-7)
    assert row.liquid_rate == pytest.approx(5253.17516889899, rel=1e-7)


# -------------------------------------------------------- solver choices
def test_gauss_and_sparse_solvers_agree():
    kwargs = dict(nx=4, ny=3, nz=2, producers=[Producer(8, "oil_rate", 200.0)],
                  injectors=[Injector(1, "water_rate", 150.0)], t_end=6.0, pc=True)
    a = Simulator(make_model(**kwargs), linear_solver="sparse").run()
    b = Simulator(make_model(**kwargs), linear_solver="gauss").run()
    assert np.max(np.abs(a.pressure - b.pressure)) < 1e-6
    assert np.max(np.abs(a.sw - b.sw)) < 1e-9


# --------------------------------------------------------- material balance
def _fluid_in_place(sim, model):
    """Total fluid in the reservoir in ft^3 at standard conditions."""
    g, pvt = model.grid, model.pvt
    phi = model.porosity * (1.0 + model.rock_compressibility * (sim.p_o - 3000.0))
    pw = sim.p_o  # no capillary pressure in these tests
    inv_bo, inv_bw = pvt.inv_bo(sim.p_o), pvt.inv_bw(pw)
    return float(np.sum(g.volume * phi * ((1 - sim.s_w) * inv_bo + sim.s_w * inv_bw)))


def test_material_balance_with_a_producer():
    """Fluid taken out equals fluid in place lost (closed box, one producer)."""
    model = make_model(nx=6, producers=[Producer(1, "oil_rate", 10.0)], dt=0.5, t_end=30.0)
    sim = Simulator(model)
    start = _fluid_in_place(sim, model)
    res = sim.run()
    w = res.wells
    produced = (w.cum_oil.iloc[-1] + w.cum_water.iloc[-1]) * BBL_TO_FT3
    lost = start - _fluid_in_place(sim, model)
    # first-order in time: 0.5 % at dt = 2 days, 0.17 % at dt = 0.5 day
    assert produced == pytest.approx(lost, rel=0.01)
    assert w.oil_rate.eq(10.0).all()


def test_material_balance_with_injector_and_producer():
    model = make_model(
        nx=8, producers=[Producer(8, "oil_rate", 10.0)],
        injectors=[Injector(1, "water_rate", 10.0)], dt=0.5, t_end=40.0,
    )
    sim = Simulator(model)
    start = _fluid_in_place(sim, model)
    res = sim.run()
    w = res.wells
    inj = w[w.kind == "injector"].cum_water.iloc[-1]
    out = w[w.kind == "producer"]
    net = (out.cum_oil.iloc[-1] + out.cum_water.iloc[-1]) - inj
    lost = start - _fluid_in_place(sim, model)
    # error well below 0.5 % of the 2250 ft3 that was injected
    assert net * BBL_TO_FT3 == pytest.approx(lost, abs=0.005 * inj * BBL_TO_FT3)


# -------------------------------------------------------------- waterflood
def test_water_front_moves_from_injector_toward_producer():
    model = make_model(
        nx=10, producers=[Producer(10, "oil_rate", 100.0, min_bhp=500.0)],
        injectors=[Injector(1, "water_rate", 100.0)], dt=0.25, t_end=60.0,
    )
    res = Simulator(model).run()
    sw_first, sw_last = res.sw[0], res.sw[-1]
    assert np.all(sw_last >= sw_first - 1e-9)
    # water saturation falls monotonically with distance from the injector
    assert np.all(np.diff(sw_last) <= 1e-4)   # tiny rises far away come from water compressibility
    assert sw_last[0] > 0.5 and sw_last[-1] < sw_last[0]
    # the saturation stays inside its physical limits
    assert res.sw.min() >= model.swc - 1e-12 and res.sw.max() <= 1 - model.sor + 1e-12


def test_gravity_segregation_is_stable():
    """A column that starts with water below oil (equilibrium) does not move."""
    model = make_model(
        nx=1, ny=1, nz=6, pc=True, dt=2.0, t_end=40.0, dz=10.0,
        initial=Equilibrium(datum_pressure=3000.0, woc_depth=60.0),
    )
    sim = Simulator(model)
    s0 = sim.s_w.copy()
    sim.run()
    assert np.max(np.abs(sim.s_w - s0)) < 1e-8
    # water saturation increases with depth
    assert np.all(np.diff(s0) >= -1e-12)


# --------------------------------------------------------------- well limits
def test_minimum_bhp_switches_producer_to_bhp_control():
    model = make_model(
        nx=4, producers=[Producer(1, "oil_rate", 40.0, min_bhp=2900.0)], dt=0.5, t_end=40.0,
    )
    res = Simulator(model).run()
    w = res.wells
    assert w["mode"].iloc[0] == "oil_rate"
    assert w["mode"].iloc[-1] == "bhp"
    after = w[w["mode"] == "bhp"]
    assert np.allclose(after.bhp, 2900.0)
    # once on bhp control the rate declines
    assert np.all(np.diff(after.oil_rate) <= 1e-9)
    assert after.oil_rate.iloc[0] <= 40.0 * 1.001
    assert (after.oil_rate >= -1e-9).all()


def test_minimum_oil_rate_stops_the_run():
    model = make_model(
        nx=4, producers=[Producer(1, "oil_rate", 40.0, min_bhp=2900.0, min_oil_rate=30.0)],
        dt=0.5, t_end=200.0,
    )
    res = Simulator(model).run()
    assert res.stop_reason is not None and "oil rate" in res.stop_reason
    assert res.wells.oil_rate.iloc[-1] <= 30.0
    assert res.times[-1] < 200.0


def test_liquid_rate_control_splits_by_mobility():
    model = make_model(
        nx=5, initial=UniformStart(3000.0, 0.5), dt=0.5, t_end=2.0,
        producers=[Producer(1, "liquid_rate", 300.0)],
    )
    res = Simulator(model).run()
    first = res.wells.iloc[0]
    assert first.liquid_rate == pytest.approx(300.0)
    assert 0.0 < first.water_cut < 1.0
    assert first.water_cut == pytest.approx(first.water_rate / 300.0)


def test_bhp_producer_rate_follows_well_equation():
    model = make_model(
        nx=3, producers=[Producer(2, "bhp", 2500.0)], dt=0.5, t_end=3.0,
        initial=UniformStart(3000.0, 0.2),
    )
    sim = Simulator(model)
    sim.run()
    w = sim.results().wells
    assert np.all(w.bhp == 2500.0)
    assert np.all(w.oil_rate > 0)
    # rate decreases as the reservoir depletes
    assert w.oil_rate.is_monotonic_decreasing


def test_injector_at_constant_bhp_raises_pressure():
    model = make_model(
        nx=5, injectors=[Injector(1, "bhp", 3600.0)], dt=0.5, t_end=10.0,
        initial=UniformStart(3000.0, 0.2),
    )
    sim = Simulator(model)
    sim.run()
    assert sim.p_o.min() > 3000.0
    w = sim.results().wells
    assert np.all(w.water_rate > 0)
    assert w.water_rate.is_monotonic_decreasing


def test_no_wells_run_is_valid():
    res = Simulator(make_model(nx=3, t_end=3.0)).run()
    assert res.wells.empty
    assert res.pressure.shape == (4, 3)
    assert np.allclose(res.pressure, 3000.0)
