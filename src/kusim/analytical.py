"""Buckley-Leverett (Welge) analytical solution for 1-D waterflooding.

Used to check the simulator against an exact answer (Chapter 6, Application 2)
and as a teaching tool.  Assumptions: incompressible fluids, no gravity, no
capillary pressure, water injected at one end of a 1-D bed, and a uniform
initial water saturation ``swi``.

The fractional flow of water is

    fw(Sw) = 1 / (1 + (kro * mu_w) / (krw * mu_o))

The solution is a shock (the "front") followed by a smooth rarefaction.  The
front saturation is found with Welge's tangent construction: draw the straight
line from (swi, fw(swi)) that touches the fw curve from above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fluids import RockFluidTable


def fractional_flow(rock_fluid: RockFluidTable, sw, mu_o, mu_w):
    """Fractional flow of water at saturation sw."""
    kro, krw = rock_fluid.relperm_oil(sw), rock_fluid.relperm_water(sw)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(krw > 0, kro * mu_w / (krw * mu_o), np.inf)
    return 1.0 / (1.0 + ratio)


@dataclass
class WelgeSolution:
    """Result of the tangent construction.

    Attributes
    ----------
    swi : float
        Initial water saturation.
    sw_front : float
        Water saturation just behind the front.
    front_speed : float
        Dimensionless front speed: front position in pore volumes (of the bed
        length) per pore volume injected = dfw/dSw at the front.
    """

    rock_fluid: RockFluidTable
    mu_o: float
    mu_w: float
    swi: float
    sw_max: float
    sw_front: float
    front_speed: float

    @property
    def breakthrough_pv(self) -> float:
        """Pore volumes of water injected when the front reaches the end of the bed."""
        return 1.0 / self.front_speed

    def fw(self, sw):
        return fractional_flow(self.rock_fluid, sw, self.mu_o, self.mu_w)

    def profile(self, x_over_length, pv_injected):
        """Water saturation at x/L (0 = injector, 1 = producer) after ``pv_injected`` PV."""
        x = np.atleast_1d(np.asarray(x_over_length, dtype=float))
        sw_grid = np.linspace(self.sw_front, self.sw_max, 4001)
        fw = self.fw(sw_grid)
        dfw = np.gradient(fw, sw_grid)
        x_of_sw = pv_injected * dfw                 # position of each saturation behind the front
        # x_of_sw decreases as sw increases; np.interp needs increasing x, so reverse
        sw_behind = np.interp(x, x_of_sw[::-1], sw_grid[::-1])
        front_x = pv_injected * self.front_speed
        out = np.where(x < front_x, sw_behind, self.swi)
        return out if out.size > 1 else float(out[0])


def welge(rock_fluid: RockFluidTable, mu_o: float, mu_w: float, swi: float, sor: float) -> WelgeSolution:
    """Welge tangent construction for the given relative-permeability table."""
    sw_max = 1.0 - sor
    sw = np.linspace(swi, sw_max, 4001)[1:]
    fw = fractional_flow(rock_fluid, sw, mu_o, mu_w)
    fw_i = float(fractional_flow(rock_fluid, swi, mu_o, mu_w))
    slope = (fw - fw_i) / (sw - swi)
    best = int(np.argmax(slope))
    return WelgeSolution(
        rock_fluid=rock_fluid, mu_o=mu_o, mu_w=mu_w, swi=swi, sw_max=sw_max,
        sw_front=float(sw[best]), front_speed=float(slope[best]),
    )
