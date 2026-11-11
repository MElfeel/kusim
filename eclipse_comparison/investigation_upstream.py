"""Why did kusim disagree with ECLIPSE on the waterflood?  Reproduces the table in findings.md.

Run from anywhere:   py eclipse_comparison/investigation_upstream.py

It reads the ECLIPSE summary in eclipse/KUSIM_BONUS.UNSMRY (needs `pip install resdata`), runs kusim on the
case that ECLIPSE ran (the numbers below are read from eclipse/KUSIM_BONUS.DATA) and prints the RMS
difference for a few combinations.  Nothing is patched: the thesis rule for the upstream block is the
simulator option ``upstream="pressure"`` (porting-notes decision 18), the default is ``"potential"``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from resdata.summary import Summary

from kusim import Simulator
from kusim.fluids import PVTTable, RockFluidTable
from kusim.grid import Grid
from kusim.model import Equilibrium, ReservoirModel
from kusim.wells import Injector, Producer

HERE = Path(__file__).resolve().parent

# --- the case in eclipse/KUSIM_BONUS.DATA ----------------------------------------------------------
SWOF = np.array([[0.20, 0.00, 0.85, 8.0], [0.40, 0.05, 0.35, 2.5], [0.60, 0.20, 0.10, 0.8],
                 [0.75, 0.40, 0.02, 0.3], [0.85, 0.60, 0.00, 0.15], [1.00, 1.00, 0.00, 0.0]])
PVDO = np.array([[300.0, 1.25, 1.0], [800.0, 1.20, 1.1], [6000.0, 1.15, 2.0]])
RHO_O, RHO_W, BW, MU_W = 49.0, 63.0, 1.02, 0.8


def pvt(eclipse_style):
    """The table as given, or resampled so that 1/B and 1/(B*mu) are linear between the nodes."""
    p, bo, mu = PVDO.T
    if not eclipse_style:
        return PVTTable(p, bo, [BW] * 3, mu, [MU_W] * 3)
    pf = np.linspace(p[0], p[-1], 571)                         # 10 psi spacing, contains the nodes
    inv_b, inv_bmu = np.interp(pf, p, 1 / bo), np.interp(pf, p, 1 / (bo * mu))
    return PVTTable(pf, 1 / inv_b, [BW] * len(pf), inv_b / inv_bmu, [MU_W] * len(pf))


def build(dt, eclipse_pvt=False):
    g = Grid([500.0] * 5, [500.0] * 5, [50.0] * 3, top_depth=8000.0)
    kx = np.repeat([200.0, 1000.0, 200.0], 25)
    g_o = RHO_O / 62.4
    return ReservoirModel(
        g, kx, 0.8 * kx, 0.1 * kx, 0.20, 4e-6, pvt(eclipse_pvt),
        RockFluidTable(SWOF[:, 0], SWOF[:, 2], SWOF[:, 1], SWOF[:, 3]), g_o, RHO_W / 62.4, 0.20, 0.15,
        Equilibrium(4500.0 - 0.433 * g_o * 8075.0, 8500.0), dt, 2000.0,
        producers=[Producer(g.column_blocks(1, 1), "liquid_rate", 10000.0, radius=0.35, min_bhp=2000.0)],
        injectors=[Injector(g.column_blocks(5, 5), "water_rate", 11000.0, radius=0.35)])


def run(upstream="potential", dt=5.0, eclipse_pvt=False):
    sim = Simulator(build(dt, eclipse_pvt), upstream=upstream)
    res = sim.run()
    pv = sim.grid.volume * sim.model.porosity
    w = res.wells
    P, I = w[w.well == "P1"].reset_index(drop=True), w[w.well == "I1"].reset_index(drop=True)
    return pd.DataFrame({"TIME": P.time, "FPRP": (res.pressure[1:] * pv).sum(axis=1) / pv.sum(),
                         "WBHP:P1": P.bhp, "WBHP:I1": I.bhp, "WOPR:P1": P.oil_rate, "WWCT:P1": P.water_cut})


def load_eclipse():
    s = Summary(str(HERE / "eclipse" / "KUSIM_BONUS"))
    df = pd.DataFrame({k: np.asarray(s.numpy_vector(k)) for k in ["TIME", "FPRP", "WBHP:P1", "WBHP:I1", "WOPR:P1", "WWCT:P1"]})
    return df[(df.TIME % 5 == 0) & (df.TIME > 0)].reset_index(drop=True)      # report steps only


def rms(k, e):
    return {v: np.sqrt(np.mean((np.interp(e.TIME, k.TIME, k[v]) - e[v]) ** 2)) for v in e.columns if v != "TIME"}


if __name__ == "__main__":
    ecl = load_eclipse()
    rows = {
        "thesis rule (upstream by pressure)": run("pressure"),
        "upstream by potential (the default)": run("potential"),
        "  + ECLIPSE-style PVT interpolation": run("potential", eclipse_pvt=True),
        "  same, dt = 1": run("potential", dt=1.0, eclipse_pvt=True),
        "  same, dt = 10": run("potential", dt=10.0, eclipse_pvt=True),
    }
    table = pd.DataFrame({name: rms(k, ecl) for name, k in rows.items()}).T
    print("RMS difference kusim - ECLIPSE over 2000 days (psi, psi, psi, STB/d, fraction)")
    print(table.round(3).to_string())
