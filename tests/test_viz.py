"""Visualization: geometry, figures, animations, interactive view and VTK export."""

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from kusim import Grid, Simulator, viz
from kusim.cases import layered_waterflood
from kusim.viz.fields import cutaway_keep, frame_indices, pick_snapshot, well_completions


@pytest.fixture(scope="module")
def results():
    model = layered_waterflood(nx=6, ny=5, nz=3, days=40.0, dt=1.0, rate=60.0)
    return Simulator(model, save_every=5).run()


@pytest.fixture(scope="module")
def results_single_layer_wells():
    """Same model, but each well is completed in one layer only."""
    model = layered_waterflood(nx=6, ny=5, nz=3, days=40.0, dt=1.0, rate=60.0, vertical_wells=False)
    return Simulator(model, save_every=5).run()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------- geometry
def test_cell_corners_reproduce_cell_volumes():
    g = Grid([100, 75, 50], [60, 40], [10, 20], top_depth=[0, 5, 10, 0, 5, 10])
    corners = g.cell_corners()
    assert corners.shape == (g.n_cells, 8, 3)
    size = corners[:, 6] - corners[:, 0]               # far bottom corner minus near top corner
    assert np.allclose(size[:, 0] * size[:, 1] * size[:, 2], g.volume)
    # the dipping top depths show up in the corners
    assert corners[g.cell_index(g.block_number(2, 1, 1)), 0, 2] == pytest.approx(5.0)
    assert np.allclose(g.cell_centres()[2], g.depth)


def test_edges():
    g = Grid([100, 75, 50], [60, 40], [10, 20], top_depth=3.0)
    assert list(g.x_edges) == [0, 100, 175, 225]
    assert list(g.y_edges) == [0, 60, 100]
    assert list(g.z_edges) == [3, 13, 33]


def test_snapshot_and_frame_helpers(results):
    assert pick_snapshot(results, -1) == len(results.times) - 1
    assert pick_snapshot(results, time=20.0) == int(np.argmin(np.abs(results.times - 20.0)))
    with pytest.raises(IndexError):
        pick_snapshot(results, 999)
    idx = frame_indices(results, max_frames=4)
    assert idx[0] == 0 and idx[-1] == len(results.times) - 1 and len(idx) <= 4


def test_well_completions(results, results_single_layer_wells):
    wells = {w["name"]: w for w in well_completions(results)}
    assert wells["I1"]["cells"] == [(1, 1, 1), (1, 1, 2), (1, 1, 3)] and wells["I1"]["kind"] == "injector"
    assert wells["P1"]["cells"] == [(6, 5, 1), (6, 5, 2), (6, 5, 3)]
    assert wells["P1"]["blocks"] == results.grid.column_blocks(6, 5)
    single = {w["name"]: w for w in well_completions(results_single_layer_wells)}
    assert single["I1"]["cells"] == [(1, 1, 3)] and single["P1"]["cells"] == [(6, 5, 1)]


def test_cutaway_removes_a_corner_only():
    g = Grid([1] * 6, [1] * 5, [1] * 3)
    keep = cutaway_keep(g)
    assert 0 < (~keep).sum() < g.n_cells // 2
    assert keep[g.cell_index(g.block_number(1, 1, 1))] and keep[g.cell_index(g.block_number(6, 5, 3))]


# ----------------------------------------------------------- still figures
def test_plot_layer_and_layers(results):
    fig = viz.plot_layer(results, "sw", layer=3)
    ax = fig.axes[0]
    assert "layer 3" in ax.get_title() and "t = 40" in ax.get_title()
    assert len(ax.lines) >= 1                                    # the injector marker
    fig2 = viz.plot_layers(results, "pressure", time=10.0)
    assert len([a for a in fig2.axes if a.get_title().startswith("Layer")]) == 3
    with pytest.raises(ValueError):
        viz.plot_layer(results, "sw", layer=4)
    with pytest.raises(ValueError):
        viz.plot_layer(results, "porosity")


def test_plot_layer_marks_wells_in_every_layer_they_are_completed_in(results):
    for layer in (1, 2, 3):                               # both wells are completed in all layers
        assert len(viz.plot_layer(results, "sw", layer=layer).axes[0].lines) == 2
    labels = [t.get_text() for t in viz.plot_layer(results, "sw", layer=2).axes[0].texts]
    assert "P1 (3 layers)" in labels and "I1 (3 layers)" in labels


def test_plot_layer_only_marks_wells_in_that_layer(results_single_layer_wells):
    res = results_single_layer_wells
    assert len(viz.plot_layer(res, "sw", layer=2).axes[0].lines) == 0
    assert len(viz.plot_layer(res, "sw", layer=1).axes[0].lines) == 1          # producer
    assert len(viz.plot_layer(res, "sw", layer=3).axes[0].lines) == 1          # injector


def test_plot_cross_section(results):
    fig = viz.plot_cross_section(results, "sw", plane="xz", index=1)
    ax = fig.axes[0]
    assert ax.yaxis_inverted()
    # the injector is in row j = 1: dashed wellbore to the top perforation, thick completed
    # interval, and the perforation squares
    assert len(ax.lines) == 3
    assert ax.lines[2].get_marker() == "s" and len(ax.lines[2].get_xdata()) == 3
    assert len(viz.plot_cross_section(results, "so", plane="yz", index=6).axes[0].lines) == 3
    assert len(viz.plot_cross_section(results, "sw", plane="xz", index=3).axes[0].lines) == 0
    with pytest.raises(ValueError):
        viz.plot_cross_section(results, "sw", plane="xy")
    with pytest.raises(ValueError):
        viz.plot_cross_section(results, "sw", plane="xz", index=99)


def test_plot_voxels(results):
    fig = viz.plot_voxels(results, "sw", cutaway=True)
    ax = fig.axes[0]
    assert ax.zaxis_inverted()
    # each well: a dashed stub and the wellbore through the perforations (plus markers)
    assert len(ax.lines) == 4
    assert len(viz.plot_voxels(results, "sw", wells=False).axes[0].lines) == 0
    viz.plot_voxels(results, "pressure", snapshot=0)


def test_plot_wells(results):
    fig = viz.plot_wells(results)
    assert len(fig.axes) == 4
    fig_p = viz.plot_wells(results, wells=["P1"])
    assert "I1" not in [t.get_text() for t in fig_p.axes[1].get_legend().get_texts()]
    with pytest.raises(ValueError):
        viz.plot_wells(results, wells=["nope"])
    viz.plot_pressure_history(results)


def test_plot_wells_without_wells():
    from conftest import make_model
    res = Simulator(make_model(nx=3, t_end=3.0)).run()
    with pytest.raises(ValueError):
        viz.plot_wells(res)
    viz.plot_layer(res, "pressure")             # maps still work with no wells


def test_flat_field_gets_a_usable_colour_range():
    from conftest import make_model
    res = Simulator(make_model(nx=3, t_end=2.0)).run()           # nothing changes: pressure constant
    fig = viz.plot_layer(res, "pressure")
    vmin, vmax = fig.axes[0].collections[0].get_clim()
    assert vmax > vmin


# ------------------------------------------------------------- animation
def test_animate_layer_writes_a_gif(results, tmp_path):
    path = viz.animate_layer(results, tmp_path / "layer.gif", layer=3, max_frames=4, fps=4)
    with Image.open(path) as gif:
        assert gif.n_frames == 4


def test_animate_voxels_writes_a_gif(results, tmp_path):
    path = viz.animate_voxels(results, tmp_path / "model.gif", max_frames=3, fps=3)
    with Image.open(path) as gif:
        assert gif.n_frames == 3


# ------------------------------------------------------------ interactive
def test_interactive_3d_structure(results, tmp_path):
    pytest.importorskip("plotly")
    fig = viz.interactive_3d(results, "sw", max_frames=5)
    mesh = fig.data[0]
    g = results.grid
    n = g.n_cells
    shell_faces = 2 * (g.nx * g.ny + g.nx * g.nz + g.ny * g.nz)      # only the outside of the box is drawn
    assert len(mesh.x) == 8 * n and len(mesh.i) == 2 * shell_faces
    assert len(fig.frames) == 5
    assert [f.name for f in fig.frames][0] == "0" and fig.frames[-1].name == "40"
    assert len(fig.data) == 5                                    # mesh + (wellbore, perforations) per well
    marks = [t for t in fig.data if t.name.endswith("perforations")]
    assert len(marks) == 2 and all(len(t.x) == 3 for t in marks)
    assert "layer 2" in marks[0].text[1]
    # frames only carry colours, one per triangle
    assert len(fig.frames[-1].data[0].intensity) == len(mesh.i)
    assert fig.layout.scene.zaxis.autorange == "reversed"
    path = viz.save_html(fig, tmp_path / "model.html")
    assert path.exists() and path.stat().st_size > 1000


def test_interactive_3d_cutaway_and_start(results):
    pytest.importorskip("plotly")
    full = viz.interactive_3d(results, "sw", max_frames=3)
    cut = viz.interactive_3d(results, "sw", max_frames=3, cutaway=True, start_at=-1)
    g = results.grid
    removed = np.flatnonzero(~cutaway_keep(g))
    used_cells = set(np.unique(np.concatenate([cut.data[0].i, cut.data[0].j, cut.data[0].k]) // 8))
    assert used_cells.isdisjoint(removed)                        # nothing of the removed cells is drawn
    assert len(cut.data[0].i) != len(full.data[0].i)
    assert np.array_equal(cut.data[0].intensity, cut.frames[-1].data[0].intensity)
    assert cut.layout.sliders[0].active == 2


# ------------------------------------------------------------------- VTK
def test_export_vtk(results, tmp_path):
    pvd = viz.export_vtk(results, tmp_path, name="run", max_frames=4)
    entries = ET.parse(pvd).getroot().findall(".//DataSet")
    assert len(entries) == 4
    assert [float(e.get("timestep")) for e in entries][-1] == results.times[-1]
    last = ET.parse(tmp_path / entries[-1].get("file")).getroot()
    piece = last.find(".//Piece")
    n = results.grid.n_cells
    assert int(piece.get("NumberOfCells")) == n and int(piece.get("NumberOfPoints")) == 8 * n
    arrays = {a.get("Name"): a.text.split() for a in last.iter("DataArray")}
    assert np.allclose(np.array(arrays["sw"], dtype=float), results.sw[-1], atol=1e-7)
    assert np.allclose(np.array(arrays["pressure"], dtype=float), results.pressure[-1], rtol=1e-7)
    assert np.allclose(np.array(arrays["so"], dtype=float), 1 - results.sw[-1], atol=1e-7)
    assert set(arrays["types"]) == {"12"}                         # all hexahedra
    assert len(arrays["Points"]) == 3 * 8 * n
    z = np.array(arrays["Points"], dtype=float)[2::3]
    assert z.max() <= 0.0 and z.min() == pytest.approx(-results.grid.z_edges[-1])   # elevation = -depth


def test_vtk_hexahedra_have_positive_volume(results, tmp_path):
    """Corner order must make each hexahedron right-handed (positive volume in VTK)."""
    path = viz.write_vtu(results, 0, tmp_path / "one.vtu")
    root = ET.parse(path).getroot()
    pts = np.array([a for a in root.find(".//Points/DataArray").text.split()], dtype=float).reshape(-1, 8, 3)
    # scalar triple product of three edges meeting at corner 0 (bottom face, then up)
    e1, e2, e3 = pts[:, 1] - pts[:, 0], pts[:, 3] - pts[:, 0], pts[:, 4] - pts[:, 0]
    assert np.all(np.einsum("ij,ij->i", np.cross(e1, e2), e3) > 0)


def test_export_vtk_writes_the_wells(results, tmp_path):
    viz.export_vtk(results, tmp_path, name="run", max_frames=2)
    root = ET.parse(tmp_path / "run_wells.vtp").getroot()
    piece = root.find(".//Piece")
    assert piece.get("NumberOfLines") == "2" and piece.get("NumberOfPoints") == "8"   # 2 wells x (wellhead + 3)
    arrays = {a.get("Name"): a.text.split() for a in root.iter("DataArray")}
    assert arrays["offsets"] == ["4", "8"]
    assert arrays["perforation"] == ["0", "1", "1", "1"] * 2
    assert sorted(arrays["kind"]) == ["0", "1"]
    z = np.array(arrays["Points"], dtype=float)[2::3]
    assert z[0] > 0 > z[1]                                       # the wellhead is above the model (elevation > 0)
    assert np.all(np.diff(z[:4]) < 0)                            # going down the well


def test_export_vtk_without_wells_writes_no_well_file(tmp_path):
    from conftest import make_model
    plain = Simulator(make_model(nx=3, t_end=3.0)).run()
    viz.export_vtk(plain, tmp_path)
    assert not (tmp_path / "kusim_wells.vtp").exists()


# ------------------------------------------------- multi-layer well profiles
def test_plot_inflow_profile_for_producer_and_injector(results):
    fig = viz.plot_inflow_profile(results, "P1")
    ax = fig.axes[0]
    assert ax.yaxis_inverted()                                    # layer 1 at the top
    assert [t.get_text() for t in ax.get_yticklabels()][0].startswith("layer 1")
    assert len(ax.patches) == 6                                   # an oil and a water bar for each of 3 layers
    shares = [t.get_text() for t in ax.texts]
    assert len(shares) == 3 and shares[0].endswith("%")
    inj = viz.plot_inflow_profile(results, "I1", time=20.0).axes[0]
    assert len(inj.patches) == 3 and "t = 20" in inj.get_title()
    with pytest.raises(ValueError):
        viz.plot_inflow_profile(results, "nope")


def test_plot_perforation_rates(results):
    fig = viz.plot_perforation_rates(results, "P1")
    assert len(fig.axes) == 2                                     # oil and water
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend == [f"layer {k} (block {b})" for k, b in zip((1, 2, 3), results.grid.column_blocks(6, 5))]
    assert len(viz.plot_perforation_rates(results, "I1").axes) == 1
    with pytest.raises(ValueError):
        viz.plot_perforation_rates(results, "X9")


def test_inflow_profile_needs_perforation_data():
    from conftest import make_model
    res = Simulator(make_model(nx=3, t_end=3.0)).run()
    with pytest.raises(ValueError):
        viz.plot_inflow_profile(res, "P1")


def test_exposed_triangles_hide_interior_faces():
    from kusim.viz.interactive import exposed_triangles
    g = Grid([1.0] * 3, [1.0] * 3, [1.0] * 3)
    keep = np.ones(g.n_cells, dtype=bool)
    tri, owner = exposed_triangles(g, keep)
    assert len(tri) == 2 * 6 * 9                                 # 6 sides of 3 x 3 faces, 2 triangles each
    centre = g.cell_index(g.block_number(2, 2, 2))
    assert centre not in set(owner)                              # the middle cell is completely hidden
    keep[centre] = False                                         # removing it exposes its six neighbours' faces
    tri2, owner2 = exposed_triangles(g, keep)
    assert len(tri2) == len(tri) + 2 * 6
    assert len(set(owner2)) == len(set(owner)) == 26             # the six face-centre cells were already on the shell
