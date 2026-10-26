"""Solving the pressure equation: a 7-diagonal linear system  M p = d.

Each cell contributes one row with at most seven non-zero entries: itself and
its six neighbours.  The original program stored the full N x N matrix and
solved it by Gaussian elimination; that is easy to read and fine for tiny
grids, but it needs N^2 memory and about N^3 operations.  For a 75-cell grid
that does not matter; for 10,000 cells it does.

Two solvers are provided:

``"sparse"``
    SciPy's sparse direct solver.  The default.  It uses only the non-zero
    entries.

``"gauss"``
    The textbook Gaussian elimination from Chapter 5 (section 5.5), written out
    in plain loops on a dense matrix, as in the FORTRAN program.  No pivoting.
    Slow, but every line corresponds to the equations in the chapter.  Use it
    to see how the algorithm works, or to check the sparse solver on small grids.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

METHODS = ("sparse", "gauss")


def solve_linear_system(matrix, rhs, method="sparse"):
    """Solve ``matrix @ x = rhs`` where ``matrix`` is a SciPy sparse matrix."""
    if method == "sparse":
        return spsolve(matrix.tocsc(), rhs)
    if method == "gauss":
        return gauss_elimination(matrix.toarray(), rhs)
    raise ValueError(f"unknown solver {method!r}; choose from {METHODS}")


def gauss_elimination(a, d):
    """Solve a x = d by forward elimination and back substitution (Chapter 5, 5.5).

    Forward elimination turns ``a`` into an upper-triangular matrix:

        for each pivot row k, for each row i below it:
            factor = -a[i, k] / a[k, k]
            row_i  = row_i + factor * row_k

    Back substitution then finds the unknowns from the last row upwards:

        x_i = (d_i - sum_{j>i} a_ij x_j) / a_ii
    """
    a = np.array(a, dtype=float)
    d = np.array(d, dtype=float)
    n = len(d)

    for k in range(n - 1):
        for i in range(k + 1, n):
            if a[i, k] == 0.0:
                continue  # nothing to eliminate (saves time on a sparse pattern)
            factor = -a[i, k] / a[k, k]
            a[i, k:] += factor * a[k, k:]
            d[i] += factor * d[k]

    x = np.zeros(n)
    x[n - 1] = d[n - 1] / a[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - a[i, i + 1:] @ x[i + 1:]) / a[i, i]
    return x
