"""Visualization: maps, 3-D views, well profiles, animations and ParaView export.

    from kusim import viz

    viz.plot_wells(results)                     # rates, pressures, water cut, cumulative
    viz.plot_inflow_profile(results, "P1")      # how a multi-layer well draws from each layer
    viz.plot_perforation_rates(results, "P1")   # the same through time
    viz.plot_layer(results, "sw", layer=1)      # map of one layer
    viz.plot_cross_section(results, "sw")       # vertical slice
    viz.plot_voxels(results, "pressure")        # 3-D blocks
    viz.animate_layer(results, "movie.gif")     # time-lapse GIF
    viz.interactive_3d(results).show()          # Plotly, with a time slider
    viz.export_vtk(results, "out/")             # open the .pvd file in ParaView

Everything except the interactive Plotly view needs only matplotlib.
"""

from .animate import animate_layer, animate_voxels
from .interactive import interactive_3d, save_html
from .maps import plot_cross_section, plot_layer, plot_layers, plot_voxels
from .vtk import export_vtk, write_vtu, write_wells_vtp
from .wells import plot_inflow_profile, plot_perforation_rates, plot_pressure_history, plot_wells

__all__ = [
    "animate_layer",
    "animate_voxels",
    "export_vtk",
    "interactive_3d",
    "plot_cross_section",
    "plot_inflow_profile",
    "plot_layer",
    "plot_layers",
    "plot_perforation_rates",
    "plot_pressure_history",
    "plot_voxels",
    "plot_wells",
    "save_html",
    "write_vtu",
    "write_wells_vtp",
]
