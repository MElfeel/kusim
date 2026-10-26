"""Wells completed in several layers."""

import numpy as np
import pytest

from kusim import Equilibrium, Grid, Injector, Producer, Simulator, UniformStart
from kusim.constants import BBL_TO_FT3
from kusim.wells import well_index

from conftest import make_model
from test_simulator import _fluid_in_place

LAYER_PERMS = np.array([100.0, 300.0, 800.0])


def layered(nx=3, ny=1, producers=(), injectors=(), **kwargs):
    k = np.repeat(LAYER_PERMS, nx * ny)
    return make_model(nx=nx, ny=ny, nz=3, k=k, producers=producers, injectors=injectors, **kwargs)


def column(nx, ny, i, j):
    return Grid([100.0] * nx, [100.0] * ny, [20.0] * 3).column_blocks(i, j)


# ---------------------------------------------------------------- set-up
def test_column_blocks_helper():
    g = Grid([1] * 4, [1] * 3, [1] * 3)
    assert g.column_blocks(2, 3) == [10, 22, 34]
    assert g.column_blocks(2, 3, layers=[2, 3]) == [22, 34]


def test_a_single_block_in_a_list_equals_a_plain_block():
    a = Simulator(make_model(nx=5, producers=[Producer(2, "oil_rate", 15.0)], t_end=10.0)).run()
    b = Simulator(make_model(nx=5, producers=[Producer([2], "oil_rate", 15.0)], t_end=10.0)).run()
    assert np.array_equal(a.pressure, b.pressure) and np.array_equal(a.sw, b.sw)
    assert a.completions == {"P1": [2]}


def test_overlapping_completions_are_rejected():
    with pytest.raises(ValueError):
        make_model(nx=3, producers=[Producer([1, 2], "bhp", 1000.0)], injectors=[Injector(2, "bhp", 4000.0)])
    with pytest.raises(ValueError):
        make_model(nx=3, producers=[Producer([1, 99], "bhp", 1000.0)])
    with pytest.raises(ValueError):
        Producer([], "bhp", 1000.0)


# ------------------------------------------------------ rate allocation
def test_rate_is_shared_in_proportion_to_flow_capacity():
    """At the first step all layers have the same mobility, so shares follow kh (well index)."""
    blocks = column(3, 1, 1, 1)
    model = layered(producers=[Producer(blocks, "oil_rate", 30.0)], t_end=1.0, initial=UniformStart(3000.0, 0.2))
    res = Simulator(model).run()
    perf = res.perforations_of("P1")
    assert list(perf.block) == blocks and list(perf.k) == [1, 2, 3]
    expected = LAYER_PERMS / LAYER_PERMS.sum() * 30.0
    assert np.allclose(perf.oil_rate, expected, rtol=1e-9)
    assert perf.oil_rate.sum() == pytest.approx(30.0)


def test_well_totals_are_the_sum_of_the_perforations():
    model = layered(producers=[Producer(column(3, 1, 1, 1), "liquid_rate", 40.0)],
                    injectors=[Injector(column(3, 1, 3, 1), "water_rate", 40.0)],
                    t_end=12.0, initial=UniformStart(3000.0, 0.3))
    res = Simulator(model).run()
    for name in ("P1", "I1"):
        total = res.well(name)
        parts = res.perforations_of(name).groupby("time")[["oil_rate", "water_rate"]].sum()
        assert np.allclose(parts.oil_rate.values, total.oil_rate.values)
        assert np.allclose(parts.water_rate.values, total.water_rate.values)
        assert (total.n_perforations == 3).all()
    assert res.well("P1").liquid_rate.eq(40.0).all() or np.allclose(res.well("P1").liquid_rate, 40.0)
    assert np.allclose(res.well("I1").water_rate, 40.0)


def test_bottom_hole_pressure_follows_from_the_rate():
    """Pwf must satisfy the well equation, Q = sum(PI * lam * (P - Pwf)), at the end of the step."""
    model = layered(producers=[Producer(column(3, 1, 2, 1), "oil_rate", 30.0)], t_end=1.0,
                    initial=UniformStart(3000.0, 0.2))
    sim = Simulator(model)
    pr = sim._fluid_properties(sim.p_o, sim.s_w)               # properties used in the step
    sim.step()
    w = sim.producers[0]
    pwf = sim.results().well("P1").bhp.iloc[0]
    q = np.sum(w.pids * pr.lam_o[w.indices] * (sim.p_o[w.indices] - w.heads - pwf))
    assert q == pytest.approx(30.0, rel=1e-9)


def test_bhp_well_draws_from_each_layer_against_one_pressure():
    blocks = column(3, 1, 1, 1)
    model = layered(producers=[Producer(blocks, "bhp", 2500.0)], t_end=1.0, dt=0.01,
                    initial=UniformStart(3000.0, 0.2))
    res = Simulator(model).run()
    first = res.perforations[res.perforations.time == res.perforations.time.min()]
    # every layer obeys q = PI * mobility * (block pressure - well pressure at its depth); mobility is the same
    # in all layers at the first step, so q / (P - Pwf) follows the layer permeabilities
    heads = Simulator(model).producers[0].heads            # wellbore pressure increase with depth
    per_drawdown = first.oil_rate.values / (first.pressure.values - 2500.0 - heads)
    assert np.allclose(per_drawdown / per_drawdown[0], LAYER_PERMS / LAYER_PERMS[0], rtol=1e-9)
    assert (first.oil_rate > 0).all()
    assert np.allclose(res.well("P1").bhp, 2500.0)


def test_injector_rate_is_shared_by_injectivity():
    model = layered(injectors=[Injector(column(3, 1, 1, 1), "water_rate", 50.0)], t_end=1.0,
                    initial=UniformStart(3000.0, 0.2))
    res = Simulator(model).run()
    perf = res.perforations_of("I1")
    assert np.allclose(perf.water_rate, LAYER_PERMS / LAYER_PERMS.sum() * 50.0, rtol=1e-9)
    # injection needs a well pressure above the (capacity-weighted) average block pressure,
    # compared at the depth of the top perforation
    heads = Simulator(model).injectors[0].heads
    assert res.well("I1").bhp.iloc[0] > np.average(perf.pressure - heads, weights=LAYER_PERMS)


def test_injector_at_constant_bhp_favours_the_permeable_layer():
    model = layered(injectors=[Injector(column(3, 1, 1, 1), "bhp", 3500.0)], t_end=4.0,
                    initial=UniformStart(3000.0, 0.2))
    res = Simulator(model).run()
    perf = res.perforations
    first = perf[perf.time == perf.time.min()].water_rate.values
    assert (first > 0).all() and first[2] > first[1] > first[0]
    assert (res.well("I1").water_rate > 0).all()
    # Without a hydrostatic head in the wellbore, the deepest perforation can end up above the
    # bottom-hole pressure once its block has filled up, and it then back-flows (cross-flow).
    # That is a known simplification, so only the well total is required to stay positive.


# --------------------------------------------------- physics and limits
def test_material_balance_with_a_multilayer_producer():
    model = layered(nx=3, producers=[Producer(column(3, 1, 1, 1), "oil_rate", 12.0)], dt=0.5, t_end=30.0)
    sim = Simulator(model)
    start = _fluid_in_place(sim, model)
    res = sim.run()
    w = res.wells
    produced = (w.cum_oil.iloc[-1] + w.cum_water.iloc[-1]) * BBL_TO_FT3
    lost = start - _fluid_in_place(sim, model)
    assert produced == pytest.approx(lost, rel=0.01)


def test_high_permeability_layer_floods_first():
    model = layered(nx=6, producers=[Producer(column(6, 1, 6, 1), "liquid_rate", 40.0)],
                    injectors=[Injector(column(6, 1, 1, 1), "water_rate", 40.0)], dt=1.0, t_end=300.0,
                    initial=UniformStart(3000.0, 0.2))
    res = Simulator(model).run()
    sw = res.sw_3d()                                  # (nz, ny, nx)
    mean_by_layer = sw.mean(axis=(1, 2))
    assert mean_by_layer[2] > mean_by_layer[1] > mean_by_layer[0]
    # and the producer's inflow shifts toward the layer that has been flooded
    perf = res.perforations_of("P1")
    last = perf[perf.time == perf.time.max()]
    assert last.water_rate.values[2] > last.water_rate.values[0]


def test_minimum_bhp_switch_works_for_a_multilayer_well():
    model = layered(producers=[Producer(column(3, 1, 1, 1), "oil_rate", 30.0, min_bhp=2900.0)],
                    dt=0.5, t_end=40.0)
    res = Simulator(model).run()
    w = res.wells
    assert w["mode"].iloc[0] == "oil_rate" and w["mode"].iloc[-1] == "bhp"
    assert np.allclose(w[w["mode"] == "bhp"].bhp, 2900.0)


def test_each_perforation_has_its_own_well_index():
    model = layered(producers=[Producer(column(3, 1, 1, 1), "bhp", 2500.0, radius=0.3, skin=1.0)])
    sim = Simulator(model)
    w = sim.producers[0]
    g = model.grid
    for pid, c in zip(w.pids, w.indices):
        assert pid == pytest.approx(well_index(model.kx[c], model.ky[c], g.dx[c], g.dy[c], g.dz[c], 0.3, 1.0))
    assert w.pids[2] / w.pids[0] == pytest.approx(LAYER_PERMS[2] / LAYER_PERMS[0])


# -------------------------------------------------------- wellbore pressure head
def test_wellbore_head_follows_the_fluid_gradient():
    model = layered(producers=[Producer(column(3, 1, 1, 1), "bhp", 2500.0)],
                    injectors=[Injector(column(3, 1, 3, 1), "bhp", 3500.0, wellbore_gravity=0.5)])
    sim = Simulator(model)
    # layers are 20 ft thick; producers default to the oil gravity (0.8), injectors to what was given
    assert np.allclose(sim.producers[0].heads, 0.433 * 0.8 * np.array([0.0, 20.0, 40.0]))
    assert np.allclose(sim.injectors[0].heads, 0.433 * 0.5 * np.array([0.0, 20.0, 40.0]))
    single = Simulator(make_model(nx=3, producers=[Producer(2, "bhp", 2500.0)]))
    assert single.producers[0].heads.tolist() == [0.0]
    off = Simulator(layered(producers=[Producer(column(3, 1, 1, 1), "bhp", 2500.0, wellbore_gravity=0.0)]))
    assert np.all(off.producers[0].heads == 0.0)


def test_no_backflow_when_the_reservoir_is_in_hydrostatic_equilibrium():
    """In equilibrium every layer has the same drawdown, so every layer produces."""
    model = make_model(nx=2, ny=1, nz=3, k=300.0, producers=[Producer(Grid([100.0] * 2, [100.0], [20.0] * 3).column_blocks(1, 1), "bhp", 2800.0)],
                       initial=Equilibrium(datum_pressure=3000.0, woc_depth=500.0), dt=0.05, t_end=0.5, pc=False)
    # make the capillary table flat so that the equilibrium saturation is the same in every layer
    res = Simulator(model).run()
    perf = res.perforations_of("P1")
    first = perf[perf.time == perf.time.min()].sort_values("k")
    well = Simulator(model).producers[0]
    drawdown = first.pressure.values - 2800.0 - well.heads
    assert np.allclose(drawdown, drawdown[0], atol=1e-6)         # the same drawdown in every layer
    assert (perf.oil_rate > 0).all()


def test_demo_case_has_no_backflow_in_the_producer():
    from kusim.cases import layered_waterflood
    res = Simulator(layered_waterflood(days=400.0), save_every=50).run()
    perf = res.perforations_of("P1")
    assert perf.oil_rate.min() >= -1e-6 and perf.water_rate.min() >= -1e-6
