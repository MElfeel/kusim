"""Example 1: a 1-D waterflood compared with the Buckley-Leverett solution.

This is Application 2 of Chapter 6: water is injected at one end of a 600 ft bed
and oil is produced at the other.  We simulate it with KUSIM and compare the
water saturation profile with the exact (Welge) answer.

Run it from the repository root:

    python examples/01_waterflood_1d.py

It saves ``waterflood_1d.png`` next to this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kusim import (Grid, Injector, Producer, PVTTable, ReservoirModel, RockFluidTable,
                   Simulator, UniformStart)
from kusim.analytical import welge

# --- data from Chapter 6, Application 2 -------------------------------------
LENGTH = 600.0            # ft between injector and producer
WIDTH, THICKNESS = 1320.0, 20.0   # ft  (cross-section area 26,400 ft^2)
POROSITY, PERMEABILITY = 0.25, 350.0
MU_O, MU_W = 2.0, 1.0     # cp
SWI = SOR = 0.2
RATE = 600.0              # STB/day injected and produced (the chapter prints 900 / 600)
N_BLOCKS = 40

# Relative permeability is not given in the chapter, so we use simple Corey curves.
sw = np.linspace(SWI, 1 - SOR, 41)
s = (sw - SWI) / (1 - SWI - SOR)
rock_fluid = RockFluidTable(sw, kro=(1 - s) ** 2, krw=0.4 * s**2, pcow=np.zeros_like(s))

# Constant formation volume factors (Bo = Bw = 1) keep the fluids incompressible,
# so that the analytical solution applies exactly.
pvt = PVTTable([1000.0, 5000.0], bo=[1, 1], bw=[1, 1], mu_o=[MU_O] * 2, mu_w=[MU_W] * 2)

pore_volume_stb = POROSITY * LENGTH * WIDTH * THICKNESS / 5.615
t_end = 0.52 * pore_volume_stb / RATE    # a little beyond 0.5 pore volumes injected

model = ReservoirModel(
    grid=Grid([LENGTH / N_BLOCKS] * N_BLOCKS, [WIDTH], [THICKNESS]),
    kx=PERMEABILITY, ky=PERMEABILITY, kz=PERMEABILITY, porosity=POROSITY,
    rock_compressibility=1e-7, pvt=pvt, rock_fluid=rock_fluid,
    gamma_o=0.85, gamma_w=1.0, swc=SWI, sor=SOR,
    initial=UniformStart(pressure=3500.0, sw=SWI),
    dt=0.5, t_end=t_end,
    injectors=[Injector(block=1, mode="water_rate", target=RATE)],
    producers=[Producer(block=N_BLOCKS, mode="liquid_rate", target=RATE)],
)

# Save a snapshot at 0.1, 0.25 and 0.5 pore volumes injected.
sim = Simulator(model)
snapshots = {}
for pv in (0.1, 0.25, 0.5):
    while sim.time * RATE / pore_volume_stb < pv - 1e-9:
        sim.step()
    snapshots[pv] = sim.s_w.copy()
results = sim.results()

# --- compare with the exact solution ----------------------------------------
exact = welge(rock_fluid, MU_O, MU_W, SWI, SOR)
x = (np.arange(N_BLOCKS) + 0.5) / N_BLOCKS
xs = np.linspace(0, 1, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for pv, sw_sim in snapshots.items():
    line, = ax1.plot(x * LENGTH, sw_sim, "o", ms=3, label=f"KUSIM, {pv} PV")
    ax1.plot(xs * LENGTH, exact.profile(xs, pv), "-", color=line.get_color(), lw=1)
ax1.set(xlabel="distance from injector (ft)", ylabel="water saturation",
        title="Saturation profile (dots: simulator, lines: Buckley-Leverett)")
ax1.legend()

w = results.well("P1")
ax2.plot(w.time * RATE / pore_volume_stb, w.water_cut, label="KUSIM")
ax2.axvline(exact.breakthrough_pv, ls="--", color="k", label="Welge breakthrough")
ax2.set(xlabel="pore volumes injected", ylabel="water cut at producer", title="Water cut")
ax2.legend()
fig.tight_layout()

out = Path(__file__).with_name("waterflood_1d.png")
fig.savefig(out, dpi=130)
print(f"Welge breakthrough at {exact.breakthrough_pv:.3f} PV; saved {out.name}")
