"""Export results as VTK files for ParaView (or any VTK viewer).

``export_vtk`` writes one ``.vtu`` file per saved time and a ``.pvd`` file that lists
them with their times.  In ParaView: File > Open the ``.pvd`` file, click Apply, pick
"sw" or "pressure" in the colour menu, and press Play for the time-lapse.

The files are written as plain text with the standard library only, so no VTK package
is needed.  Every block is a hexahedron with its own eight corners, which keeps
dipping layers exact.  Depth is written as elevation (z = -depth), so the model
is the right way up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..results import Results
from .fields import frame_indices, well_completions

_VTK_HEXAHEDRON = 12


def _array(name, values, dtype="Float64", components=1):
    flat = " ".join(f"{v:.8g}" if dtype == "Float64" else str(int(v)) for v in np.ravel(values))
    comp = f' NumberOfComponents="{components}"' if components > 1 else ""
    return f'<DataArray type="{dtype}" Name="{name}"{comp} format="ascii">{flat}</DataArray>'


def write_vtu(results: Results, snapshot: int, path):
    """Write one snapshot to a ``.vtu`` file."""
    g = results.grid
    n = g.n_cells
    corners = g.cell_corners()
    # VTK hexahedron: bottom face first (counter-clockwise seen from above), then the top face.
    ordered = np.concatenate([corners[:, 4:8, :], corners[:, 0:4, :]], axis=1).copy()
    ordered[:, :, 2] *= -1.0                              # depth -> elevation
    points = ordered.reshape(-1, 3)
    connectivity = np.arange(n * 8)
    offsets = np.arange(1, n + 1) * 8
    types = np.full(n, _VTK_HEXAHEDRON)

    sw = results.sw[snapshot]
    cell_arrays = [
        _array("pressure", results.pressure[snapshot]),
        _array("sw", sw),
        _array("so", 1.0 - sw),
        _array("block", np.arange(1, n + 1), "Int64"),
        _array("i", g.i + 1, "Int64"),
        _array("j", g.j + 1, "Int64"),
        _array("k", g.k + 1, "Int64"),
    ]
    xml = "\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "<UnstructuredGrid>",
        f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{n}">',
        "<CellData>", *cell_arrays, "</CellData>",
        "<Points>", _array("Points", points, components=3), "</Points>",
        "<Cells>",
        _array("connectivity", connectivity, "Int64"),
        _array("offsets", offsets, "Int64"),
        _array("types", types, "UInt8"),
        "</Cells>",
        "</Piece>", "</UnstructuredGrid>", "</VTKFile>",
    ])
    Path(path).write_text(xml)
    return Path(path)


def write_wells_vtp(results: Results, path, stub=0.35):
    """Write the wellbores as polylines (a ``.vtp`` file) with the perforations as points.

    Each well is one polyline from above the top of the model down through the centres of its
    perforated blocks.  Cell data: ``well`` (1, 2, ... in the order of ``results.completions``)
    and ``kind`` (0 = producer, 1 = injector).  Point data: ``perforation`` (1 at a perforated
    block, 0 at the wellhead point) and ``block`` (block number, 0 at the wellhead).
    """
    g = results.grid
    cx, cy, cz = g.cell_centres()
    top = g.z_edges[0] - stub * (g.z_edges[-1] - g.z_edges[0])
    points, connectivity, offsets, perforation, block, well_id, kind = [], [], [], [], [], [], []
    for number, w in enumerate(well_completions(results), start=1):
        cells = [g.cell_index(b) for b in w["blocks"]]
        members = [(cx[cells[0]], cy[cells[0]], top, 0, 0)] + [
            (cx[c], cy[c], cz[c], 1, c + 1) for c in cells]
        for x, y, z, is_perf, b in members:
            connectivity.append(len(points))
            points.append((x, y, -z))                          # depth -> elevation
            perforation.append(is_perf)
            block.append(b)
        offsets.append(len(connectivity))
        well_id.append(number)
        kind.append(0 if w["kind"] == "producer" else 1)
    n_points, n_lines = len(points), len(offsets)
    xml = "\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        "<PolyData>",
        f'<Piece NumberOfPoints="{n_points}" NumberOfVerts="0" NumberOfLines="{n_lines}" '
        'NumberOfStrips="0" NumberOfPolys="0">',
        "<PointData>", _array("perforation", perforation, "Int64"), _array("block", block, "Int64"),
        "</PointData>",
        "<CellData>", _array("well", well_id, "Int64"), _array("kind", kind, "Int64"), "</CellData>",
        "<Points>", _array("Points", points, components=3), "</Points>",
        "<Lines>", _array("connectivity", connectivity, "Int64"), _array("offsets", offsets, "Int64"),
        "</Lines>",
        "</Piece>", "</PolyData>", "</VTKFile>",
    ])
    Path(path).write_text(xml)
    return Path(path)


def export_vtk(results: Results, folder, name="kusim", max_frames=100):
    """Write ``<name>_<n>.vtu`` for each snapshot, ``<name>.pvd`` and ``<name>_wells.vtp``.

    Returns the ``.pvd`` path.  If the run saved more than ``max_frames`` snapshots, evenly
    spaced ones are written.  Open both the ``.pvd`` (the reservoir over time) and the
    ``_wells.vtp`` (the wellbores and perforations) in ParaView; use the "Tube" filter on the
    wells to make them thick.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, s in enumerate(frame_indices(results, max_frames)):
        file_name = f"{name}_{number:04d}.vtu"
        write_vtu(results, s, folder / file_name)
        entries.append(f'<DataSet timestep="{results.times[s]:g}" group="" part="0" file="{file_name}"/>')
    pvd = "\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "<Collection>", *entries, "</Collection>", "</VTKFile>",
    ])
    path = folder / f"{name}.pvd"
    path.write_text(pvd)
    if not results.wells.empty:
        write_wells_vtp(results, folder / f"{name}_wells.vtp")
    return path
