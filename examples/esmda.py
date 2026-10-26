"""Ensemble Smoother with Multiple Data Assimilation (ES-MDA) on the kusim history-matching API.

This is NOT part of the kusim package (like ``enkf_reference.py``, it is a reference that shows the interface is enough).
The idea in plain words: run every model over the *whole* history, compare with *all* the data, and improve the uncertain
parameters (here the permeability) a little.  Then run again from day 0 and repeat.  Each pass pretends that the data are
noisier than they are, by a factor ``alpha`` on the variance, so that each step is small and careful.  The factors satisfy

    sum over the passes of 1 / alpha  =  1,

so that together the passes use the data exactly once.  The permeability does not change in reality; only our knowledge of
it does, and the models are always re-run from time zero, so they are consistent with the parameters by construction.

The update of one pass is the stochastic EnKF analysis of ``enkf_reference.enkf_analysis`` with the measurement errors
multiplied by ``sqrt(alpha)``, applied to the parameters only (the data of all dates stacked into one vector).
"""

import numpy as np

from enkf_reference import enkf_analysis
from kusim.history import data_matrix


def esmda_alphas(n_passes=4):
    """Equal inflation factors for ``n_passes`` passes: ``alpha = n_passes`` each, so that sum(1/alpha) = 1."""
    return [float(n_passes)] * n_passes


def check_alphas(alphas):
    """Raise ValueError unless sum(1 / alpha) = 1 (the data are used exactly once over all the passes)."""
    total = sum(1.0 / a for a in alphas)
    if not np.isclose(total, 1.0):
        raise ValueError(f"the inflation factors must satisfy sum(1/alpha) = 1, got {total:.4f}")


def stack_data(observations, dates, records):
    """The observations of all ``dates`` as one vector, and the simulated data of the members as one matrix.

    Returns ``(d_obs, sigma, D)`` with shapes (n_data,), (n_data,), (n_data, members); the order is date by date."""
    d_obs, sigma, rows = [], [], []
    for date in dates:
        names, values, sig = observations.at(date)
        d_obs.append(values)
        sigma.append(sig)
        rows.append(data_matrix(records, date, names))
    return np.concatenate(d_obs), np.concatenate(sigma), np.vstack(rows)


def esmda_update(Y, D, d_obs, sigma, alpha, rng):
    """One ES-MDA pass: the parameters ``Y`` (n_parameters x members) updated with the data of the whole history.

    ``D`` are the simulated data (n_data x members), ``d_obs`` and ``sigma`` the observed data and their errors.
    Each member sees its own noisy copy of the data, with an error inflated by ``sqrt(alpha)``.

    There is no localisation: the distance localisation of the EnKF example (``enkf_twin.localisation_yd``, applied to the
    state-data covariance only) made this update overshoot when the data of all dates are stacked (see ``docs/history-matching.md``)."""
    return enkf_analysis(Y, D, d_obs, np.asarray(sigma) * np.sqrt(alpha), rng)
