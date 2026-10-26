"""Interactive 3-D time-lapse with Plotly: rotate, zoom, hover, and drag a time slider.

Needs the optional ``plotly`` package (``pip install plotly``).  In a Jupyter
notebook just put the figure on the last line of a cell.  Outside a notebook,
``save_html`` writes a standalone web page you can open in any browser or send
to someone.
"""

from __future__ import annotations

import numpy as np

from ..results import Results
from .fields import FIELDS, color_limits, cutaway_keep, field_data, frame_indices, well_completions

# The six faces of a hexahedral cell, from the corner order of ``Grid.cell_corners``
# (corners 0-3: top face, 4-7: bottom face; 0 = (x0, y0), 1 = (x1, y0), 2 = (x1, y1), 3 = (x0, y1)).
# Each face is two triangles, tagged with the direction in which the neighbouring cell lies.
_FACES = {
    "z_minus": ((0, 1, 2), (0, 2, 3)),     # top
    "z_plus": ((4, 6, 5), (4, 7, 6)),      # bottom
    "y_minus": ((0, 1, 5), (0, 5, 4)),
    "x_plus": ((1, 2, 6), (1, 6, 5)),
    "y_plus": ((2, 3, 7), (2, 7, 6)),
    "x_minus": ((3, 0, 4), (3, 4, 7)),
}


def exposed_triangles(grid, keep):
    """Triangles of the faces that can be seen: the outer shell and any cut surfaces.

    A face between two drawn cells is hidden inside the model, so it is left out.  This keeps the
    figure clean when it is semi-transparent and makes the file much smaller.

    Returns ``(triangles, owner)``: an array (n, 3) of vertex indices (8 vertices per cell, in cell
    order) and, for each triangle, the cell it belongs to.
    """
    triangles, owner = [], []
    for direction, pair in _FACES.items():
        neighbour = grid.neighbour[direction]
        visible = keep & (~grid.is_open[direction] | ~keep[neighbour])
        cells = np.flatnonzero(visible)
        for corners in pair:
            triangles.append(cells[:, None] * 8 + np.array(corners)[None, :])
            owner.append(cells)
    return np.concatenate(triangles), np.concatenate(owner)


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:                              # pragma: no cover
        raise ImportError("interactive plots need plotly: pip install plotly") from exc
    return go


def interactive_3d(results: Results, field="sw", max_frames=60, opacity=1.0, vmin=None, vmax=None,
                   vertical_exaggeration=3.0, colorscale=None, show_wells=True, cutaway=False,
                   start_at=0, title=None):
    """A Plotly figure of the 3-D model with a time slider and play button.

    Every block is a coloured box; only the faces you can see (the outside and any cut surface)
    are drawn.  The colour scale is fixed over the whole run.

    Parameters
    ----------
    vertical_exaggeration : float
        Stretches depth (layers are usually thin compared with their extent).
    cutaway : bool
        Remove one corner of the model so the inside can be seen.
    opacity : float
        1 is solid; try 0.4-0.6 to see the wells through the blocks.
    start_at : int
        Which frame is shown first (0 = the start of the run, -1 = the end).
    """
    go = _import_plotly()
    g = results.grid
    lo, hi = color_limits(results, field, vmin, vmax)
    data = field_data(results, field)
    frames_idx = frame_indices(results, max_frames)

    keep = cutaway_keep(g) if cutaway else np.ones(g.n_cells, dtype=bool)
    vertices = g.cell_corners().reshape(-1, 3)                    # 8 vertices per cell
    triangles, owner = exposed_triangles(g, keep)

    def cell_colors(snapshot):
        # the value of the cell each visible triangle belongs to, rounded to keep the file small
        return np.round(data[snapshot][owner], 5)

    def mesh(snapshot):
        return go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            intensity=cell_colors(snapshot), intensitymode="cell",
            colorscale=colorscale or _PLOTLY_SCALES.get(field, "Viridis"), cmin=lo, cmax=hi,
            opacity=opacity, flatshading=True,
            colorbar=dict(title=FIELDS[field].label, x=1.0, len=0.7),
            hovertemplate=FIELDS[field].label + ": %{intensity:.4g}<extra></extra>",
            name=field,
        )

    start = frames_idx[start_at]
    traces = [mesh(start)]
    if show_wells:
        cx, cy, cz = g.cell_centres()
        top = g.z_edges[0] - 0.35 * (g.z_edges[-1] - g.z_edges[0])   # the wellhead stands above the model
        for w in well_completions(results):
            cells = [g.cell_index(b) for b in w["blocks"]]
            color = "red" if w["kind"] == "producer" else "blue"
            # wellbore: from the wellhead down through every perforated block
            traces.append(go.Scatter3d(
                x=[cx[cells[0]], *cx[cells]], y=[cy[cells[0]], *cy[cells]], z=[top, *cz[cells]],
                mode="lines+text", line=dict(color=color, width=7),
                text=[w["name"]] + [""] * len(cells), textposition="top center",
                name=w["name"], hoverinfo="name",
            ))
            # one marker per perforation, with the block and layer in the hover text
            traces.append(go.Scatter3d(
                x=cx[cells], y=cy[cells], z=cz[cells], mode="markers",
                marker=dict(size=5, color=color, symbol="square", line=dict(color="white", width=1)),
                text=[f"{w['name']}: perforation in block {b} (layer {c[2]})" for b, c in zip(w["blocks"], w["cells"])],
                hoverinfo="text", name=f"{w['name']} perforations",
            ))

    # A frame carries only the new colours; the geometry stays as in the first trace.
    frames = [go.Frame(data=[go.Mesh3d(intensity=cell_colors(s))], traces=[0], name=f"{results.times[s]:g}")
              for s in frames_idx]
    steps = [dict(method="animate", label=f.name,
                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for f in frames]
    lx, ly = g.x_edges[-1], g.y_edges[-1]
    lz = (g.z_edges[-1] - g.z_edges[0]) * vertical_exaggeration
    longest = max(lx, ly, lz)
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=title or f"{FIELDS[field].label}",
        scene=dict(
            xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="depth (ft)",
            zaxis=dict(autorange="reversed"),
            aspectmode="manual", aspectratio=dict(x=lx / longest, y=ly / longest, z=lz / longest),
        ),
        showlegend=False,
        sliders=[dict(steps=steps, active=frames_idx.index(start), x=0.15, len=0.8,
                      currentvalue=dict(prefix="time (days): "), pad=dict(t=30))],
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0, xanchor="left", yanchor="top",
                          pad=dict(t=30, r=10),
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=150, redraw=True),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0),
                                                           transition=dict(duration=0))])])],
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


_PLOTLY_SCALES = {"pressure": "Viridis", "sw": "Blues", "so": "YlOrBr"}


def save_html(fig, path, offline=False):
    """Write the figure to a standalone HTML file.

    With ``offline=True`` the Plotly library is embedded (about 4 MB larger) so the
    file works without an internet connection.
    """
    fig.write_html(str(path), include_plotlyjs=True if offline else "cdn")
    return path
