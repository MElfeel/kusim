"""The reservoir grid: a 3-D Cartesian block grid stored as flat arrays.

Cell numbering follows the thesis and the FORTRAN code.  Blocks are numbered
from 1: along x first, then y, then z (layer by layer, top layer first).

    block = (k-1)*NX*NY + (j-1)*NX + i          (i, j, k start at 1)

Inside the Python code every cell array is a 1-D NumPy array of length
``n_cells`` indexed from 0, so cell number ``block`` lives at index
``block - 1``.  Use ``grid.block_number(i, j, k)`` when you want the thesis
numbering and ``grid.to_3d(array)`` to see an array as ``(nz, ny, nx)``.

Grid sizes follow the original program: ``dx`` is one value per column of
blocks, ``dy`` one per row, ``dz`` one per layer.
"""

from __future__ import annotations

import numpy as np

DIRECTIONS = ("x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus")


class Grid:
    """Block sizes, volumes, depths and neighbour connections.

    Parameters
    ----------
    dx, dy, dz : sequences of float
        Block size in ft: ``dx`` has NX values, ``dy`` has NY values and
        ``dz`` has NZ values (one per layer).
    top_depth : float or array, optional
        Depth (ft, below the reference datum) of the top of layer 1.  One
        number for a flat structure, or NX*NY numbers (one per column of
        blocks) for a dipping one.  This is the ``DH`` input of the original
        program.
    """

    def __init__(self, dx, dy, dz, top_depth=0.0):
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        dz = np.asarray(dz, dtype=float)
        self.nx, self.ny, self.nz = len(dx), len(dy), len(dz)
        self.n_layer = self.nx * self.ny
        self.n_cells = self.n_layer * self.nz

        # Position of each cell: i fastest, then j, then k.
        k, j, i = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self.i, self.j, self.k = i.ravel(), j.ravel(), k.ravel()

        # Block sizes and volume of every cell.
        self.dx = dx[self.i]
        self.dy = dy[self.j]
        self.dz = dz[self.k]
        self.volume = self.dx * self.dy * self.dz

        # Depth of the cell centre below the datum (called DDH in the thesis).
        top = np.asarray(top_depth, dtype=float).reshape(-1)
        if top.size == 1:
            top = np.full(self.n_layer, top[0])
        elif top.size != self.n_layer:
            raise ValueError(f"top_depth must be one number or {self.n_layer} values, got {top.size}")
        thickness_above = np.concatenate(([0.0], np.cumsum(dz)[:-1]))
        self.depth = top[self.j * self.nx + self.i] + thickness_above[self.k] + self.dz / 2

        self._build_neighbours()

    def _build_neighbours(self):
        """For each of the six directions: neighbour index and 'is there one?'.

        At the edge of the grid there is no neighbour: ``is_open`` is False and
        ``neighbour`` points back to the cell itself (so indexing is always
        safe).  Zero transmissibility on closed faces gives the no-flow
        boundary of the original program.
        """
        idx = np.arange(self.n_cells)
        step = {"x": 1, "y": self.nx, "z": self.n_layer}
        position = {"x": self.i, "y": self.j, "z": self.k}
        size = {"x": self.nx, "y": self.ny, "z": self.nz}
        self.neighbour = {}
        self.is_open = {}
        for direction in DIRECTIONS:
            axis, sign = direction[0], (-1 if direction.endswith("minus") else 1)
            moved = position[axis] + sign
            open_ = (moved >= 0) & (moved < size[axis])
            self.is_open[direction] = open_
            self.neighbour[direction] = np.where(open_, idx + sign * step[axis], idx)

    # ------------------------------------------------------------------
    # Geometry for plotting
    # ------------------------------------------------------------------
    @property
    def x_edges(self):
        """Block edges along x (length NX+1), starting at 0 ft."""
        return np.concatenate(([0.0], np.cumsum(self.dx[: self.nx])))

    @property
    def y_edges(self):
        """Block edges along y (length NY+1), starting at 0 ft."""
        return np.concatenate(([0.0], np.cumsum(self.dy[:: self.nx][: self.ny])))

    @property
    def z_edges(self):
        """Layer boundaries (length NZ+1): depth in ft below the datum, using the mean top depth."""
        thickness = self.dz[:: self.n_layer][: self.nz]
        top = float(np.mean(self.depth[: self.n_layer] - self.dz[: self.n_layer] / 2))
        return top + np.concatenate(([0.0], np.cumsum(thickness)))

    def cell_corners(self):
        """Corners of every cell as an array of shape (n_cells, 8, 3): columns are x, y, depth.

        Corner order is the usual hexahedron one: the four corners of the top face
        (counter-clockwise seen from above), then the four corners of the bottom
        face in the same order.  Each cell uses its own depth, so dipping layers
        are drawn correctly.
        """
        x0 = self.x_edges[self.i]
        y0 = self.y_edges[self.j]
        z_top = self.depth - self.dz / 2
        x1, y1, z_bot = x0 + self.dx, y0 + self.dy, z_top + self.dz
        xs = np.stack([x0, x1, x1, x0, x0, x1, x1, x0], axis=1)
        ys = np.stack([y0, y0, y1, y1, y0, y0, y1, y1], axis=1)
        zs = np.stack([z_top] * 4 + [z_bot] * 4, axis=1)
        return np.stack([xs, ys, zs], axis=2)

    def cell_centres(self):
        """Cell centre coordinates as arrays (x, y, depth), each of length n_cells."""
        return (
            self.x_edges[self.i] + self.dx / 2,
            self.y_edges[self.j] + self.dy / 2,
            self.depth,
        )

    # ------------------------------------------------------------------
    @property
    def shape(self):
        """Grid shape as (nz, ny, nx)."""
        return (self.nz, self.ny, self.nx)

    def block_number(self, i, j, k):
        """Thesis block number (starting at 1) of block (i, j, k) (each from 1)."""
        if not (1 <= i <= self.nx and 1 <= j <= self.ny and 1 <= k <= self.nz):
            raise ValueError(f"block ({i}, {j}, {k}) is outside the {self.nx}x{self.ny}x{self.nz} grid")
        return (k - 1) * self.n_layer + (j - 1) * self.nx + i

    def column_blocks(self, i, j, layers=None):
        """Block numbers of the vertical column at (i, j), top layer first.

        ``layers`` is an optional list of layer numbers (from 1), for example ``[2, 3]``
        to complete a well in the lower two layers only.  The default is every layer.
        """
        layers = range(1, self.nz + 1) if layers is None else list(layers)
        return [self.block_number(i, j, k) for k in layers]

    def block_ijk(self, block):
        """Inverse of ``block_number``: returns (i, j, k), each from 1."""
        index = self.cell_index(block)
        return int(self.i[index]) + 1, int(self.j[index]) + 1, int(self.k[index]) + 1

    def cell_index(self, block):
        """Array index (from 0) of a thesis block number (from 1)."""
        if not 1 <= block <= self.n_cells:
            raise ValueError(f"block {block} is outside 1..{self.n_cells}")
        return block - 1

    def to_3d(self, array):
        """View a flat cell array as (nz, ny, nx)."""
        return np.asarray(array).reshape(self.shape)

    def as_cell_array(self, value, name="value"):
        """Turn a single number or a per-cell list into a float array of length n_cells."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_cells, float(arr))
        arr = arr.reshape(-1)
        if arr.size != self.n_cells:
            raise ValueError(f"{name} must be one number or {self.n_cells} values, got {arr.size}")
        return arr.copy()
