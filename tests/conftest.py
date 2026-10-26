"""Shared helpers for the tests."""

from pathlib import Path

import numpy as np
import pytest

from kusim import (
    Equilibrium,
    Grid,
    Injector,
    Producer,
    PVTTable,
    ReservoirModel,
    RockFluidTable,
    UniformStart,
    read_legacy_input,
)

REPO = Path(__file__).resolve().parents[1]
LEGACY_INPUT = REPO / "legacy" / "fortran-3d3" / "INPUT.DAT"


@pytest.fixture
def legacy_model():
    return read_legacy_input(LEGACY_INPUT)


def simple_tables(krw_zero=False, pc=True):
    """A small PVT table and Corey-type relative permeability table."""
    pvt = PVTTable(
        pressure=[1000.0, 3000.0, 5000.0],
        bo=[1.30, 1.25, 1.20],
        bw=[1.02, 1.015, 1.01],
        mu_o=[1.0, 1.0, 1.0],
        mu_w=[0.5, 0.5, 0.5],
    )
    sw = np.linspace(0.2, 0.8, 7)
    s = (sw - 0.2) / 0.6
    kro = (1 - s) ** 2
    krw = np.zeros_like(s) if krw_zero else 0.4 * s**2
    pcow = 5.0 * (1 - s) if pc else np.zeros_like(s)
    return pvt, RockFluidTable(sw, kro, krw, pcow)


def make_model(nx=5, ny=1, nz=1, producers=(), injectors=(), dt=1.0, t_end=20.0,
               cr=3e-6, krw_zero=False, pc=False, initial=None, k=200.0, dx=100.0, dz=20.0):
    pvt, rf = simple_tables(krw_zero=krw_zero, pc=pc)
    grid = Grid([dx] * nx, [100.0] * ny, [dz] * nz)
    return ReservoirModel(
        grid=grid, kx=k, ky=k, kz=k, porosity=0.2, rock_compressibility=cr,
        pvt=pvt, rock_fluid=rf, gamma_o=0.8, gamma_w=1.0, swc=0.2, sor=0.2,
        initial=initial or UniformStart(pressure=3000.0, sw=0.2),
        dt=dt, t_end=t_end, producers=list(producers), injectors=list(injectors),
    )
