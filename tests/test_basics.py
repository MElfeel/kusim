"""Tables, grid, wells and the legacy input reader."""

import math

import numpy as np
import pytest

from kusim import Grid, read_legacy_input
from kusim.tables import Table, invert_decreasing
from kusim.wells import equivalent_radius, well_index

from conftest import LEGACY_INPUT


# ---------------------------------------------------------------- tables
def test_table_lookup_and_flat_extrapolation():
    t = Table([1.0, 2.0, 4.0], [10.0, 20.0, 60.0])
    assert t.lookup(1.5) == pytest.approx(15.0)
    assert t.lookup(3.0) == pytest.approx(40.0)
    assert t.lookup(0.0) == 10.0        # below the table: first value
    assert t.lookup(9.0) == 60.0        # above the table: last value


def test_table_slope_uses_segment_containing_x():
    t = Table([1.0, 2.0, 4.0], [10.0, 20.0, 60.0])
    assert t.slope(1.5) == pytest.approx(10.0)
    assert t.slope(3.0) == pytest.approx(20.0)
    assert t.slope(2.0) == pytest.approx(20.0)   # exactly on a node: segment to the right
    assert t.slope(0.0) == pytest.approx(10.0)   # outside: end segments
    assert t.slope(9.0) == pytest.approx(20.0)


def test_table_rejects_bad_input():
    with pytest.raises(ValueError):
        Table([1.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        Table([1.0], [0.0])


def test_invert_decreasing_with_flat_region():
    sw = [0.0, 0.1, 0.2, 0.5, 1.0]
    pc = [30.0, 30.0, 20.0, 5.0, 0.0]
    assert invert_decreasing(sw, pc, 12.5) == pytest.approx(0.35)
    assert invert_decreasing(sw, pc, 0.0) == pytest.approx(1.0)
    assert invert_decreasing(sw, pc, 99.0) == pytest.approx(0.0)  # above the table
    # Round trip through the forward table
    forward = Table(sw[1:], pc[1:])
    assert invert_decreasing(sw, pc, forward.lookup(0.3)) == pytest.approx(0.3)


# ------------------------------------------------------------------ grid
def test_block_numbering_matches_thesis():
    g = Grid([100, 75, 50, 75, 100], [100, 75, 50, 75, 100], [20, 30, 40])
    assert g.n_cells == 75 and g.shape == (3, 5, 5)
    assert g.block_number(1, 1, 1) == 1
    assert g.block_number(5, 1, 1) == 5
    assert g.block_number(1, 2, 1) == 6
    assert g.block_number(3, 3, 1) == 13      # the producer of Application 1
    assert g.block_number(1, 1, 2) == 26
    assert g.block_number(5, 5, 3) == 75
    assert g.block_ijk(13) == (3, 3, 1)
    with pytest.raises(ValueError):
        g.block_number(6, 1, 1)


def test_sizes_volumes_and_depths():
    g = Grid([100, 75, 50, 75, 100], [100, 75, 50, 75, 100], [20, 30, 40], top_depth=5.0)
    c = g.cell_index(13)
    assert (g.dx[c], g.dy[c], g.dz[c]) == (50, 50, 20)
    assert g.volume[c] == 50 * 50 * 20
    # centre of layer 1, layer 2, layer 3 below a top at 5 ft
    assert g.depth[g.cell_index(1)] == pytest.approx(5 + 10)
    assert g.depth[g.cell_index(26)] == pytest.approx(5 + 20 + 15)
    assert g.depth[g.cell_index(51)] == pytest.approx(5 + 50 + 20)


def test_neighbours_and_closed_faces():
    g = Grid([1, 1, 1], [1, 1], [1, 1])
    assert g.n_cells == 12
    c = g.cell_index(g.block_number(2, 1, 1))
    assert g.neighbour["x_plus"][c] == c + 1
    assert g.neighbour["x_minus"][c] == c - 1
    assert g.neighbour["y_plus"][c] == c + 3
    assert g.neighbour["z_plus"][c] == c + 6
    assert not g.is_open["y_minus"][c] and not g.is_open["z_minus"][c]
    # a closed face points back at the cell itself
    assert g.neighbour["y_minus"][c] == c
    # each interior face is open from both sides
    for plus, minus in (("x_plus", "x_minus"), ("y_plus", "y_minus"), ("z_plus", "z_minus")):
        assert g.is_open[plus].sum() == g.is_open[minus].sum()


def test_cell_array_helper():
    g = Grid([1, 1], [1], [1])
    assert list(g.as_cell_array(3)) == [3.0, 3.0]
    assert list(g.as_cell_array([1, 2])) == [1.0, 2.0]
    with pytest.raises(ValueError):
        g.as_cell_array([1, 2, 3])


# ----------------------------------------------------------------- wells
def test_equivalent_radius_isotropic_square_block():
    # For kx = ky and dx = dy, re = 0.28*sqrt(2)*dx/2 = 0.198*dx (Peaceman)
    assert equivalent_radius(100, 100, 50, 50) == pytest.approx(0.28 * math.sqrt(2) * 50 / 2)


def test_well_index_value():
    # Application 1, block 13: kx=ky=375, 50 x 50 x 20 ft, rw = 0.25, skin 0.5
    re = 0.28 * math.sqrt(2) * 50 / 2
    expected = 0.00708 * 375 * 20 / (math.log(re / 0.25) + 0.5)
    assert well_index(375, 375, 50, 50, 20, 0.25, 0.5) == pytest.approx(expected)


# ---------------------------------------------------------- legacy input
def test_read_legacy_input_matches_the_file():
    m = read_legacy_input(LEGACY_INPUT)
    g = m.grid
    assert g.shape == (3, 5, 5)
    assert list(g.dx[:5]) == [100, 75, 50, 75, 100]
    assert list(g.dz[::25]) == [20, 30, 40]
    assert np.all(m.kx == 375) and np.all(m.kz == 100) and np.all(m.porosity == 0.2)
    assert (m.gamma_o, m.gamma_w, m.swc, m.sor) == (0.85, 1.0, 0.2, 0.2)
    assert m.rock_compressibility == 0.00045
    assert (m.dt, m.t_end) == (1.0, 5000.0)
    assert len(m.pvt.pressure) == 4 and len(m.rock_fluid.sw) == 23
    assert m.initial.datum_pressure == 4800 and m.initial.woc_depth == 250
    assert len(m.producers) == 1 and len(m.injectors) == 0
    p = m.producers[0]
    assert (p.block, p.mode, p.target, p.radius, p.skin) == (13, "oil_rate", 5000.0, 0.25, 0.5)
    assert (p.min_bhp, p.min_oil_rate) == (2500.0, 100.0)
