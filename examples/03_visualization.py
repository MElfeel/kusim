"""Example 3: every way KUSIM can show a simulation.

Runs a small layered waterflood (``kusim.cases.layered_waterflood``).  Both wells are
vertical and completed in all three layers, and the permeability increases with depth, so
the layers take very different shares of the flow.  Writes into ``examples/output/``:

    wells.png            rates, bottom-hole pressures, water cut, cumulative volumes
    inflow_profile.png   how the producer and injector share their rate between the layers
    perforation_rates.png  the same through time (stacked by layer)
    layers_sw.png        water saturation, all three layers side by side
    section_sw.png       vertical cross-section along the injector, with its perforations
    voxels_sw.png        the whole model as 3-D blocks, wells drawn on top
    layer3_sw.gif        time-lapse of the bottom layer
    model_sw.gif         time-lapse of the 3-D model
    model_sw.html        interactive 3-D view with a time slider (open it in a browser)
    paraview/            .pvd + .vtu (reservoir) and _wells.vtp (wellbores) for ParaView

Run it from the repository root:

    python examples/03_visualization.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kusim import Simulator, viz
from kusim.cases import layered_waterflood

out = Path(__file__).with_name("output")
out.mkdir(exist_ok=True)

model = layered_waterflood()
results = Simulator(model, save_every=20).run()      # keep a snapshot every 20 days
print(f"{len(results.times)} snapshots, t = 0 ... {results.times[-1]:g} days")

# --- still pictures ----------------------------------------------------------
viz.plot_wells(results).savefig(out / "wells.png", dpi=120)
viz.plot_layers(results, "sw").savefig(out / "layers_sw.png", dpi=120)

# --- wells completed in several layers -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharex=False)
viz.plot_inflow_profile(results, "P1", time=100.0, ax=axes[0])
viz.plot_inflow_profile(results, "P1", time=1000.0, ax=axes[1])
viz.plot_inflow_profile(results, "I1", time=1000.0, ax=axes[2])
fig.tight_layout()
fig.savefig(out / "inflow_profile.png", dpi=120)
viz.plot_perforation_rates(results, "P1").savefig(out / "perforation_rates.png", dpi=120)

# The injector is at (i=1, j=1), the producer at (i=10, j=10): the diagonal is not an
# axis-aligned plane, so show the section along the injector row (j = 1).  The thick line is
# the completed interval and each square is a perforated block.
viz.plot_cross_section(results, "sw", plane="xz", index=1, vertical_exaggeration=2).savefig(
    out / "section_sw.png", dpi=120)
viz.plot_voxels(results, "sw", cutaway=True).savefig(out / "voxels_sw.png", dpi=120)
plt.close("all")

# --- time-lapses -------------------------------------------------------------
viz.animate_layer(results, out / "layer3_sw.gif", field="sw", layer=3)
viz.animate_voxels(results, out / "model_sw.gif", field="sw", cutaway=True, max_frames=26)

# --- interactive and ParaView ------------------------------------------------
# opacity < 1 lets you see the wellbores and perforations through the blocks
fig = viz.interactive_3d(results, "sw", cutaway=True, opacity=0.6)
viz.save_html(fig, out / "model_sw.html")
viz.export_vtk(results, out / "paraview")

print("wrote:", ", ".join(sorted(p.name for p in out.iterdir())))
