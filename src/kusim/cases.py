"""Ready-made reservoir models for examples, exercises and tests."""

from __future__ import annotations

import numpy as np

from .fluids import PVTTable, RockFluidTable
from .grid import Grid
from .model import ReservoirModel, UniformStart
from .wells import Injector, Producer


def corey_rock_fluid(swc=0.2, sor=0.2, krw_max=0.4, kro_max=1.0, nw=2.0, no=2.0, pc_max=0.0, points=21):
    """Relative permeability and capillary pressure tables from Corey-type power laws.

    ``kro = kro_max * (1 - s)**no`` and ``krw = krw_max * s**nw`` with the normalised
    saturation ``s = (Sw - swc) / (1 - swc - sor)``.  A non-zero ``pc_max`` (psi) gives
    a capillary pressure falling linearly from ``pc_max`` at ``swc`` to zero at ``1 - sor``.
    """
    sw = np.linspace(swc, 1.0 - sor, points)
    s = (sw - swc) / (1.0 - swc - sor)
    return RockFluidTable(sw, kro_max * (1 - s) ** no, krw_max * s**nw, pc_max * (1 - s))


def simple_pvt(mu_o=1.5, mu_w=0.5):
    """A smooth, slightly compressible black-oil PVT table (pressure 1000-5000 psi)."""
    return PVTTable(
        pressure=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        bo=[1.300, 1.285, 1.270, 1.255, 1.240],
        bw=[1.030, 1.027, 1.024, 1.021, 1.018],
        mu_o=[mu_o] * 5,
        mu_w=[mu_w] * 5,
    )


def layered_waterflood(nx=10, ny=10, nz=3, size=50.0, thickness=20.0,
                       permeability=(100.0, 300.0, 800.0), kv_over_kh=0.1,
                       rate=150.0, producer_bhp=2500.0, days=1000.0, dt=1.0, porosity=0.2,
                       vertical_wells=True):
    """A layered waterflood: an injector at one corner and a producer at the opposite corner.

    Permeability increases with depth (one value per layer, the list is repeated or
    cut to ``nz`` layers), so water runs ahead in the lower layers.  Water is injected at
    ``rate`` STB/day and the producer holds ``producer_bhp`` psi.

    With ``vertical_wells=True`` (the default) both wells are completed in every layer, so the
    rate of each layer follows its permeability.  With ``vertical_wells=False`` the injector is
    completed in the bottom layer only and the producer in the top layer only.
    """
    perm_layers = np.resize(np.asarray(permeability, dtype=float), nz)
    kh = np.repeat(perm_layers, nx * ny)
    grid = Grid([size] * nx, [size] * ny, [thickness] * nz)
    if vertical_wells:
        producer_block = grid.column_blocks(nx, ny)
        injector_block = grid.column_blocks(1, 1)
    else:
        producer_block = grid.block_number(nx, ny, 1)
        injector_block = grid.block_number(1, 1, nz)
    return ReservoirModel(
        grid=grid, kx=kh, ky=kh, kz=kh * kv_over_kh, porosity=porosity,
        rock_compressibility=4e-6,
        pvt=simple_pvt(), rock_fluid=corey_rock_fluid(),
        gamma_o=0.85, gamma_w=1.0, swc=0.2, sor=0.2,
        initial=UniformStart(pressure=3000.0, sw=0.2),
        dt=dt, t_end=days,
        injectors=[Injector(injector_block, "water_rate", rate, radius=0.3)],
        producers=[Producer(producer_block, "bhp", producer_bhp, radius=0.3)],
    )
