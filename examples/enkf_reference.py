"""A compact reference Ensemble Kalman Filter analysis step, to exercise the kusim history-matching API.

This is NOT part of the kusim package: the filter (and its localisation) is research code, and this file only
shows that ``kusim.parameters``, ``kusim.history`` and ``Simulator.set_state`` are enough to write it.  It follows
the stochastic EnKF of ``enkf/HA_localised_enkf.docx``:

    y_j^a = y_j^f + rho_YD o C_YD (rho_DD o C_DD + C_e)^-1 (d_obs,j - d_j^f)

where ``y`` is the augmented state (parameters and dynamic variables), ``d`` the simulated data, ``d_obs,j`` the
observations perturbed with the measurement error, ``C`` ensemble covariances and ``rho`` an optional
localisation (correlation) matrix applied with the Schur (element-wise) product ``o``.
"""

import numpy as np


def enkf_analysis(Y, D, d_obs, sigma, rng, rho_yd=None, rho_dd=None):
    """Update the ensemble ``Y`` (n_state x Ne) with the data.

    D       simulated data of the members, shape (n_data, Ne)
    d_obs   observed data, shape (n_data,)
    sigma   standard deviations of the measurement errors, shape (n_data,)
    rho_yd  optional localisation of the state-data covariance, shape (n_state, n_data)
    rho_dd  optional localisation of the data-data covariance, shape (n_data, n_data)
    """
    n_data, ne = D.shape
    dY = Y - Y.mean(axis=1, keepdims=True)
    dD = D - D.mean(axis=1, keepdims=True)
    c_yd = dY @ dD.T / (ne - 1)
    c_dd = dD @ dD.T / (ne - 1)
    if rho_yd is not None:
        c_yd = rho_yd * c_yd
    if rho_dd is not None:
        c_dd = rho_dd * c_dd
    perturbed = d_obs[:, None] + sigma[:, None] * rng.standard_normal((n_data, ne))
    gain_times_innovation = c_yd @ np.linalg.solve(c_dd + np.diag(sigma**2), perturbed - D)
    return Y + gain_times_innovation


def gaspari_cohn(distance, length):
    """The fifth-order Gaspari-Cohn correlation function: 1 at distance 0, 0 beyond ``2 * length``."""
    z = np.abs(np.asarray(distance, dtype=float)) / length
    out = np.zeros_like(z)
    near = z <= 1.0
    far = (z > 1.0) & (z < 2.0)
    zn, zf = z[near], z[far]
    out[near] = -0.25 * zn**5 + 0.5 * zn**4 + 0.625 * zn**3 - (5.0 / 3.0) * zn**2 + 1.0
    out[far] = (zf**5 / 12.0 - 0.5 * zf**4 + 0.625 * zf**3 + (5.0 / 3.0) * zf**2 - 5.0 * zf + 4.0 - (2.0 / 3.0) / zf)
    return out


def log_field_covariance(grid, std, range_xy, range_z):
    """Covariance matrix of the log-permeability of the blocks (exponential correlation ``exp(-d)``, with ``d`` the
    distance scaled by the ranges, and standard deviation ``std`` in ln units): a prior for a small grid (dense matrix)."""
    x, y, z = grid.cell_centres()
    d = np.sqrt(((x[:, None] - x[None, :]) / range_xy) ** 2 + ((y[:, None] - y[None, :]) / range_xy) ** 2
                + ((z[:, None] - z[None, :]) / range_z) ** 2)
    return std**2 * np.exp(-d) + 1e-9 * np.eye(grid.n_cells)


def correlated_log_field(grid, mean_by_layer, std, range_xy, range_z, rng, n=1):
    """Gaussian random fields of log-permeability on a grid (see ``log_field_covariance``), for building a prior.

    The mean is ``log(mean_by_layer[k])`` in layer k.  Returns an array (n_cells, n).  Small grids only: the
    covariance matrix is factorised directly."""
    chol = np.linalg.cholesky(log_field_covariance(grid, std, range_xy, range_z))
    mean = np.log(np.asarray(mean_by_layer, dtype=float))[grid.k]
    return mean[:, None] + chol @ rng.standard_normal((grid.n_cells, n))
