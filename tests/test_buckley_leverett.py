"""Compare the simulator with the Buckley-Leverett (Welge) analytical solution."""

import numpy as np
import pytest

from kusim import (Grid, Injector, Producer, PVTTable, ReservoirModel, RockFluidTable,
                   Simulator, UniformStart)
from kusim.analytical import fractional_flow, welge

# Application 2 of Chapter 6 (1-D, 600 ft, k = 350 md, mu_o = 2, mu_w = 1, Swi = Sor = 0.2)
SWI = SOR = 0.2
MU_O, MU_W = 2.0, 1.0


def corey_table(points=13):
    sw = np.linspace(SWI, 1 - SOR, points)
    s = (sw - SWI) / (1 - SWI - SOR)
    return RockFluidTable(sw, kro=(1 - s) ** 2, krw=0.4 * s**2, pcow=np.zeros_like(s))


def build(n_blocks, dt):
    length, area_width, thickness = 600.0, 1320.0, 20.0
    # Bo = Bw = 1 and a nearly constant PVT table: incompressible, so STB = rb
    pvt = PVTTable([1000.0, 5000.0], bo=[1.0, 1.0], bw=[1.0, 1.0], mu_o=[MU_O] * 2, mu_w=[MU_W] * 2)
    grid = Grid([length / n_blocks] * n_blocks, [area_width], [thickness])
    pore_volume_stb = 0.25 * length * area_width * thickness / 5.615
    rate = 600.0
    model = ReservoirModel(
        grid=grid, kx=350.0, ky=350.0, kz=350.0, porosity=0.25, rock_compressibility=1e-7,
        pvt=pvt, rock_fluid=corey_table(), gamma_o=0.85, gamma_w=1.0, swc=SWI, sor=SOR,
        initial=UniformStart(pressure=3500.0, sw=SWI), dt=dt, t_end=0.9 * pore_volume_stb / rate * 1.0,
        injectors=[Injector(1, "water_rate", rate)],
        producers=[Producer(n_blocks, "liquid_rate", rate)],
    )
    return model, pore_volume_stb, rate


def test_welge_construction_is_self_consistent():
    table = corey_table(points=401)   # fine table: no kinks near the front
    sol = welge(table, MU_O, MU_W, SWI, SOR)
    assert SWI < sol.sw_front < 1 - SOR
    # tangent property: the chord slope equals the local slope of the fw curve at the front
    h = 1e-4
    local = (sol.fw(sol.sw_front + h) - sol.fw(sol.sw_front - h)) / (2 * h)
    assert local == pytest.approx(sol.front_speed, rel=0.02)
    assert fractional_flow(table, SWI, MU_O, MU_W) == pytest.approx(0.0, abs=1e-12)


def test_simulated_breakthrough_matches_welge():
    n = 40
    model, pv_stb, rate = build(n, dt=0.5)
    sol = welge(corey_table(), MU_O, MU_W, SWI, SOR)
    res = Simulator(model).run()
    w = res.wells[res.wells.well == "P1"]
    pv_injected = w.time * rate / pv_stb
    # first time the produced water cut exceeds 2 %
    breakthrough = pv_injected[w.water_cut > 0.02].iloc[0]
    # numerical dispersion smears the front, so breakthrough comes a little early
    assert breakthrough == pytest.approx(sol.breakthrough_pv, rel=0.15)
    assert breakthrough < sol.breakthrough_pv * 1.02


def test_simulated_saturation_profile_follows_welge():
    n = 40
    model, pv_stb, rate = build(n, dt=0.5)
    model.t_end = 0.25 * pv_stb / rate           # stop at 0.25 PV injected
    sol = welge(corey_table(), MU_O, MU_W, SWI, SOR)
    res = Simulator(model).run()
    x = (np.arange(n) + 0.5) / n
    exact = sol.profile(x, pv_injected=0.25)
    simulated = res.sw[-1]
    # average error over the bed is small even though the front is smeared
    assert np.mean(np.abs(simulated - exact)) < 0.04
    # and the water behind the front is in the right place: same mean saturation
    assert simulated.mean() == pytest.approx(exact.mean(), abs=0.01)
