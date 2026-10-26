"""Upstream weighting by potential (porting-notes decision 18).

The thesis and the original FORTRAN choose the upstream block of a phase by comparing pressures.
With gravity the right test is the sign of the potential difference, (p_nb - p) - gamma * 0.433 * (z_nb - z).
Both rules agree when the depths are equal, and differ in vertical flow.
"""

import numpy as np
import pytest

from kusim import Injector, Producer, Simulator
from kusim.constants import PSI_PER_FT

from conftest import make_model


def _water_over_oil_column(upstream, days=300):
    """Closed column of six 40 ft blocks, water-rich on top of oil-rich, hydrostatic in oil, no wells."""
    model = make_model(nx=1, ny=1, nz=6, pc=True, dt=1.0, t_end=days, dz=40.0, k=500.0)
    sim = Simulator(model, upstream=upstream)
    sim.p_o = 3000.0 + PSI_PER_FT * model.gamma_o * (model.grid.depth - model.grid.depth[0])
    sim.s_w = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])
    sim.run()
    return sim


def test_water_sinks_through_oil_with_potential_upstream():
    sim = _water_over_oil_column("potential")
    assert sim.s_w[0] < 0.45 and sim.s_w[-1] > 0.6          # the column has turned over
    assert np.all(np.diff(sim.s_w) >= -0.02)                # and is graded: more water further down


def test_water_stays_on_top_with_pressure_upstream():
    """Documents the legacy behaviour that was corrected: the heavy water cannot move down."""
    sim = _water_over_oil_column("pressure")
    assert sim.s_w[0] > 0.7 and sim.s_w[-1] < 0.25


def test_column_turnover_keeps_the_total_water():
    """The water only moves between the blocks of the closed column (equal volumes, so the mean Sw is kept)."""
    sim = _water_over_oil_column("potential")
    assert abs(sim.s_w.mean() - 0.5) < 0.01


def test_both_rules_agree_in_horizontal_flow():
    """No depth differences, so the gravity term is zero and the two rules are the same statement."""
    def run(upstream):
        model = make_model(nx=8, dt=1.0, t_end=60.0, pc=True,
                           injectors=[Injector(1, "water_rate", 80.0)],
                           producers=[Producer(8, "bhp", 2500.0)])
        return Simulator(model, upstream=upstream).run()
    a, b = run("potential"), run("pressure")
    assert np.array_equal(a.pressure, b.pressure) and np.array_equal(a.sw, b.sw)


def test_both_rules_agree_without_vertical_flow():
    """With kz = 0 no fluid crosses between layers, so the vertical test is never used."""
    def run(upstream):
        model = make_model(nx=4, nz=3, dt=2.0, t_end=60.0, dz=40.0,
                           injectors=[Injector([1, 5, 9], "water_rate", 60.0)],
                           producers=[Producer([4, 8, 12], "bhp", 2500.0)])
        model.kz = np.zeros_like(model.kz)
        with np.errstate(divide="ignore"):
            return Simulator(model, upstream=upstream).run()
    a, b = run("potential"), run("pressure")
    assert np.array_equal(a.sw, b.sw)


def test_upstream_hook_can_be_overridden_in_a_subclass():
    """The decision is one small method (used in the bonus notebook as an exercise)."""
    class ThesisRule(Simulator):
        def _neighbour_is_upstream(self, p_phase, gamma, head, nb):
            return p_phase[nb] >= p_phase

    model = make_model(nx=1, ny=1, nz=6, pc=True, dt=1.0, t_end=50.0, dz=40.0, k=500.0)
    results = []
    for sim in (ThesisRule(model), Simulator(model, upstream="pressure")):
        sim.p_o = 3000.0 + PSI_PER_FT * model.gamma_o * (model.grid.depth - model.grid.depth[0])
        sim.s_w = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])
        sim.run()
        results.append(sim.s_w.copy())
    assert np.array_equal(*results)


def test_unknown_upstream_rule_is_rejected():
    with pytest.raises(ValueError, match="upstream"):
        Simulator(make_model(), upstream="average")
