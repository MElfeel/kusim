"""Table lookup by linear interpolation (the FORTRAN subroutine ``INTERP``).

PVT data and rock-fluid data come as tables.  The simulator needs the value
read from the table *and* its slope (for example d(1/Bo)/dP in the storage
coefficients), so a table here can return both.

Behaviour at the ends of the table (Chapter 5, section 15):

* below the first x value the result is the first y value,
* above the last x value the result is the last y value,
* in between, straight-line interpolation.

The slope is the slope of the table segment the point falls in.  Outside the
table it is the slope of the first or last segment, as in the original code.
"""

from __future__ import annotations

import numpy as np


class Table:
    """A piecewise-linear function y(x) defined by table points.

    ``x`` must be increasing.  Both ``lookup`` and ``slope`` accept a single
    number or a NumPy array.
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be one-dimensional and the same length")
        if len(self.x) < 2:
            raise ValueError("a table needs at least two points")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("table x values must be strictly increasing")

    def lookup(self, x):
        """Interpolated y at x (flat extrapolation outside the table)."""
        return np.interp(x, self.x, self.y)

    def slope(self, x):
        """Slope dy/dx of the table segment containing x."""
        x = np.asarray(x, dtype=float)
        seg = np.searchsorted(self.x, x, side="right") - 1
        seg = np.clip(seg, 0, len(self.x) - 2)
        return (self.y[seg + 1] - self.y[seg]) / (self.x[seg + 1] - self.x[seg])


def invert_decreasing(x_table, y_table, y):
    """Find x for a given y when y *decreases* as x increases.

    Used once, at initialization, to turn the capillary-pressure table
    Pcow(Sw) (Pcow falls as Sw rises) into Sw(Pcow).  This is the ``AA=0``
    branch of the FORTRAN ``INTERP``.  Outside the table the end value is used.
    Flat parts of the table (equal y values) are allowed.
    """
    x_table = np.asarray(x_table, dtype=float)
    y_table = np.asarray(y_table, dtype=float)
    if np.any(np.diff(y_table) > 0):
        raise ValueError("y must not increase with x for this function")
    # np.interp wants an increasing "x" axis, so reverse both arrays.
    return np.interp(y, y_table[::-1], x_table[::-1])
