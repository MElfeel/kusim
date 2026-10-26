"""Reading the original KUSIM ``INPUT.DAT`` file.

The FORTRAN program read its input with list-directed ``READ(13,*)`` statements,
so the file is *positional*: the program does not look at the labels, it just
reads the next numbers in a fixed order.  Rules that this reader copies:

* A read that needs numbers skips blank lines, takes numbers from a line from
  its start (text after the last number is ignored) and moves on to the next
  line if it still needs more numbers.
* A read with nothing to read (a bare ``READ(13,*)``, used to skip a label line)
  consumes exactly one line, blank or not.

The order of the data is the order of the ``READ`` statements in
``3DTP(11).FOR`` (lines 54-121).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fluids import PVTTable, RockFluidTable
from .grid import Grid
from .model import Equilibrium, ReservoirModel
from .wells import Injector, Producer


class _Records:
    """Line-by-line reader that mimics FORTRAN list-directed input."""

    def __init__(self, text):
        self.lines = text.replace("\r", "").split("\n")
        self.pos = 0

    def skip(self):
        """A bare READ: consume one line."""
        self.pos += 1

    @staticmethod
    def _leading_numbers(line):
        values = []
        for token in line.replace(",", " ").split():
            try:
                values.append(float(token))
            except ValueError:
                break
        return values

    def numbers(self, count):
        """Read ``count`` numbers.  With ``count == 0`` this consumes one line."""
        if count == 0:
            self.skip()
            return []
        out = []
        while len(out) < count:
            if self.pos >= len(self.lines):
                raise ValueError("INPUT.DAT ended while more numbers were expected")
            out.extend(self._leading_numbers(self.lines[self.pos]))
            self.pos += 1
        return out[:count]

    def one(self):
        return self.numbers(1)[0]


def read_legacy_input(path) -> ReservoirModel:
    """Read an original ``INPUT.DAT`` and return a ``ReservoirModel``.

    Differences from the FORTRAN, all deliberate:

    * The control flags ``PBC`` and ``IBC`` apply to every well in the file, as in
      the original, and become the control mode of each well.
    * ``RDATUM`` is read but not used (the original does not use it either).
    * ``DH`` is the depth of the top of layer 1 below the datum, one value per
      column of blocks.
    """
    rec = _Records(Path(path).read_text())
    rec.skip()                                          # title line
    nx, ny, nz = int(rec.one()), int(rec.one()), int(rec.one())
    n, layer = nx * ny * nz, nx * ny
    dx = rec.numbers(nx)
    dy = rec.numbers(ny)
    dz = rec.numbers(nz)
    kx, ky, kz = rec.numbers(n), rec.numbers(n), rec.numbers(n)
    phi = rec.numbers(n)
    rec.skip(); rec.one()                               # label, RDATUM (unused)
    rec.skip(); woc = rec.one()                         # label, WOCD
    rec.skip(); datum_pressure = rec.one()              # label, DPRESS
    rec.skip(); dh = rec.numbers(layer)                 # label, DH
    gamma_o, gamma_w = rec.one(), rec.one()
    swc, sor, cr = rec.one(), rec.one(), rec.one()
    dt, t_end = rec.one(), rec.one()

    n_pvt = int(rec.one())
    rec.skip()                                          # column headings
    pvt_rows = np.array([rec.numbers(5) for _ in range(n_pvt)])
    pvt = PVTTable(*pvt_rows.T)

    n_sat, pc_mult = rec.numbers(2)
    rec.skip()
    sat_rows = np.array([rec.numbers(4) for _ in range(int(n_sat))])
    rock_fluid = RockFluidTable(*sat_rows.T, pc_multiplier=pc_mult)

    rec.skip()                                          # "PRODUCTION WELLS" banner
    n_prod = int(rec.one())
    p_loc = rec.numbers(n_prod)
    p_rw = rec.numbers(n_prod)
    p_skin = rec.numbers(n_prod)
    p_min_bhp = rec.numbers(n_prod)
    p_min_rate = rec.numbers(n_prod)
    pbc = int(rec.one())
    rec.skip(); p_qo = rec.numbers(n_prod)
    rec.skip(); p_ql = rec.numbers(n_prod)
    rec.skip(); p_bhp = rec.numbers(n_prod)

    rec.skip()                                          # "INJECTION WELLS" banner
    n_inj = int(rec.one())
    i_loc = rec.numbers(n_inj)
    i_rw = rec.numbers(n_inj)
    i_skin = rec.numbers(n_inj)
    ibc = int(rec.one())
    rec.skip(); i_qw = rec.numbers(n_inj)
    rec.skip(); i_bhp = rec.numbers(n_inj)

    p_mode = {1: ("oil_rate", p_qo), 2: ("liquid_rate", p_ql), 3: ("bhp", p_bhp)}[pbc]
    producers = [
        Producer(int(p_loc[i]), p_mode[0], p_mode[1][i], p_rw[i], p_skin[i], p_min_bhp[i], p_min_rate[i])
        for i in range(n_prod)
    ]
    i_mode = {1: ("water_rate", i_qw), 2: ("bhp", i_bhp)}[ibc]
    injectors = [
        Injector(int(i_loc[i]), i_mode[0], i_mode[1][i], i_rw[i], i_skin[i]) for i in range(n_inj)
    ]

    return ReservoirModel(
        grid=Grid(dx, dy, dz, top_depth=dh),
        kx=kx, ky=ky, kz=kz, porosity=phi,
        rock_compressibility=cr,
        pvt=pvt,
        rock_fluid=rock_fluid,
        gamma_o=gamma_o, gamma_w=gamma_w,
        swc=swc, sor=sor,
        initial=Equilibrium(datum_pressure=datum_pressure, woc_depth=woc),
        dt=dt, t_end=t_end,
        producers=producers, injectors=injectors,
    )
