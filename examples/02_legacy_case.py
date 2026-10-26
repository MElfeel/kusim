"""Example 2: run the original thesis input file (Application 1) and plot it.

The 5 x 5 x 3 model in ``legacy/fortran-3d3/INPUT.DAT`` has one producer at block 13
(the middle of the top layer) taking 5000 STB/day.  With the time step of 1 day
stored in that file the explicit saturation update is unstable (the well takes
more than twice the pore volume of its own block each day), so here we use a
smaller step.  Change ``DT`` to 1.0 and watch what happens: this is the
"IMPES needs small time steps" warning of Chapter 5 in action.

Run it from the repository root:

    python examples/02_legacy_case.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kusim import Simulator, read_legacy_input

DT = 0.05            # days
T_END = 60.0         # days

root = Path(__file__).resolve().parents[1]
model = read_legacy_input(root / "legacy" / "fortran-3d3" / "INPUT.DAT")
model.dt, model.t_end = DT, T_END

results = Simulator(model, save_every=20).run()
print(f"ran {len(results.wells)} steps; stop reason: {results.stop_reason}")

well = results.well("P1")
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

ax = axes[0, 0]
ax.plot(well.time, well.oil_rate, label="oil")
ax.plot(well.time, well.water_rate, label="water")
ax.set(xlabel="time (days)", ylabel="rate (STB/day)", title="Producer rates")
ax.legend()

ax = axes[0, 1]
ax.plot(well.time, well.bhp)
ax.axhline(model.producers[0].min_bhp, ls="--", color="k", lw=0.8)
ax.set(xlabel="time (days)", ylabel="bottom-hole pressure (psi)",
       title="Bottom-hole pressure (dashed: minimum)")

for ax, field, title in ((axes[1, 0], results.pressure_3d(), "Oil pressure (psi), layer 1"),
                         (axes[1, 1], results.sw_3d(), "Water saturation, layer 1")):
    image = ax.imshow(field[0], origin="lower", cmap="viridis")
    ax.set(title=title + f", t = {results.times[-1]:g} d", xlabel="i", ylabel="j")
    fig.colorbar(image, ax=ax)

fig.tight_layout()
out = Path(__file__).with_name("legacy_case.png")
fig.savefig(out, dpi=130)
print("saved", out.name)
