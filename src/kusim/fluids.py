"""Fluid and rock-fluid property tables (PVT, relative permeability, capillary pressure).

Two tables describe the fluids:

``PVTTable``
    Oil and water formation volume factors (Bo, Bw) and viscosities against
    pressure.  The simulator works with the *reciprocal* 1/B because that is
    what appears in the equations (mobility = kr / (mu * B) and the storage
    coefficients use d(1/B)/dP).  This is why the original program turns
    ``BO`` into ``1/BO`` right after reading the table.

``RockFluidTable``
    Oil and water relative permeability and oil-water capillary pressure
    against water saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tables import Table, invert_decreasing


@dataclass
class PVTTable:
    """PVT data against pressure (psi): Bo, Bw in rb/STB, viscosities in cp."""

    pressure: np.ndarray
    bo: np.ndarray
    bw: np.ndarray
    mu_o: np.ndarray
    mu_w: np.ndarray

    def __post_init__(self):
        self.pressure = np.asarray(self.pressure, dtype=float)
        self.bo = np.asarray(self.bo, dtype=float)
        self.bw = np.asarray(self.bw, dtype=float)
        self.mu_o = np.asarray(self.mu_o, dtype=float)
        self.mu_w = np.asarray(self.mu_w, dtype=float)
        self._inv_bo = Table(self.pressure, 1.0 / self.bo)
        self._inv_bw = Table(self.pressure, 1.0 / self.bw)
        self._mu_o = Table(self.pressure, self.mu_o)
        self._mu_w = Table(self.pressure, self.mu_w)

    def inv_bo(self, p):
        """1/Bo at pressure p (STB/rb)."""
        return self._inv_bo.lookup(p)

    def d_inv_bo(self, p):
        """d(1/Bo)/dP at pressure p."""
        return self._inv_bo.slope(p)

    def inv_bw(self, p):
        return self._inv_bw.lookup(p)

    def d_inv_bw(self, p):
        return self._inv_bw.slope(p)

    def viscosity_oil(self, p):
        return self._mu_o.lookup(p)

    def viscosity_water(self, p):
        return self._mu_w.lookup(p)


@dataclass
class RockFluidTable:
    """Relative permeabilities and capillary pressure against Sw.

    ``pc_multiplier`` scales the whole capillary-pressure column (the
    ``PCMULT`` input of the original program; use 0 to switch capillary
    pressure off).
    """

    sw: np.ndarray
    kro: np.ndarray
    krw: np.ndarray
    pcow: np.ndarray
    pc_multiplier: float = 1.0

    def __post_init__(self):
        self.sw = np.asarray(self.sw, dtype=float)
        self.kro = np.asarray(self.kro, dtype=float)
        self.krw = np.asarray(self.krw, dtype=float)
        self.pcow = np.asarray(self.pcow, dtype=float) * self.pc_multiplier
        self._kro = Table(self.sw, self.kro)
        self._krw = Table(self.sw, self.krw)
        self._pcow = Table(self.sw, self.pcow)

    def relperm_oil(self, sw):
        return self._kro.lookup(sw)

    def relperm_water(self, sw):
        return self._krw.lookup(sw)

    def pcow_at(self, sw):
        """Capillary pressure Pcow = Po - Pw (psi) at water saturation sw."""
        return self._pcow.lookup(sw)

    def d_pcow(self, sw):
        """dPcow/dSw (psi) at water saturation sw."""
        return self._pcow.slope(sw)

    def sw_from_pcow(self, pcow):
        """Water saturation that has capillary pressure ``pcow`` (inverse lookup).

        Capillary pressure falls as Sw rises, so this is the inverse used to set
        the initial saturation from the height above the water-oil contact.
        """
        return invert_decreasing(self.sw, self.pcow, pcow)
