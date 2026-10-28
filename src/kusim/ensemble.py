"""Ensemble history matching: a prior ensemble, and two ways to update it with data, ES-MDA and the EnKF.

``kusim ensemble`` (``kusim.cli``) is a thin layer over ``run_esmda`` and ``run_enkf`` here. Both work on the history
matching interface of the package (``ParameterSet``, ``run_ensemble``, ``Observations``, ``SimState``), and a model is
never edited in place.

* **ES-MDA** (Ensemble Smoother with Multiple Data Assimilation): run every member over the whole history, compare with
  *all* the data, update the parameters a little, run again from day 0, and repeat. Each pass pretends the data are
  noisier by a factor ``alpha`` on the variance, so that every step is small; the factors satisfy ``sum(1 / alpha) = 1``,
  so that the passes use the data exactly once. The models are always run from time zero, so they are consistent with
  their parameters by construction. This is the method of the course (Week 5).
* **EnKF** (Ensemble Kalman Filter): assimilate the data date by date. Forecast the members to the next date (each from
  its own restart state), update the augmented state [parameters, pressure, Sw], edit the models with the updated
  parameters and restart them from the updated pressure and saturation. The static and dynamic variables are updated
  separately and are not consistent afterwards, so the members are run again from time zero at the end: that replay is
  what the reported posterior fit refers to. Optional Gaspari-Cohn localisation (``localise_length``) around the wells.

The update of both is the stochastic EnKF analysis ``enkf_analysis``:

    y_j^a = y_j^f + rho_YD o C_YD (rho_DD o C_DD + C_e)^-1 (d_obs,j - d_j^f)

with ``y`` the vector being updated, ``d`` the simulated data, ``d_obs,j`` the observations perturbed with the
measurement error, ``C`` ensemble covariances and ``rho`` an optional localisation (Schur product ``o``).

The **prior** (``sample_prior``) is Gaussian in the transformed space of the ``ParameterSet`` (``ln k`` for a log
parameter), per parameter: a mean (default: the value in the model), a standard deviation and, for a field, an optional
exponential correlation with horizontal and vertical ranges. Correlated fields use a dense covariance matrix, so they
are limited to ``MAX_CORRELATED_CELLS`` blocks. The random numbers come from independent streams of the seed: member j
of a larger ensemble is member j of a smaller one.

Limits: a twin experiment recovers the permeability field only partly (see ``docs/history-matching.md``): the data fit
improves, the field less reliably. Read ``rms`` as "does the model reproduce the history", not as "is the field right".
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Optional, Sequence

import numpy as np

from .history import data_matrix, misfit, rms, run_ensemble, run_to
from .parameters import FieldParameter, ParameterSet, _forward
from .spec import _PRIOR_ENTRY_KEYS

MAX_CORRELATED_CELLS = 3000


# ---------------------------------------------------------------------- the analysis step
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


def localisation_yd(model, data_vectors, length, n_blocks=3):
    """Distance-based localisation of the state-data covariance (Gaspari-Cohn, support ``2 * length`` ft).

    The state is ``n_blocks`` blocks of one value per grid block (for example log kx, pressure, Sw); each datum belongs to
    a well, and the weight of a block is the correlation function of its horizontal distance to the first block of that
    well (a datum of the whole field, such as ``FOPT``, is not localised). Returns rho_yd, shape (n_blocks * n_cells, n_data)."""
    grid = model.grid
    x, y, _ = grid.cell_centres()
    wells = {f"P{n}": w for n, w in enumerate(model.producers, 1)} | {f"I{n}": w for n, w in enumerate(model.injectors, 1)}
    columns = []
    for vector in data_vectors:
        well = wells.get(vector.partition(":")[2])
        if well is None:
            columns.append(np.ones(n_blocks * grid.n_cells))
            continue
        c = grid.cell_index(well.blocks[0])
        distance = np.hypot(x - x[c], y - y[c])
        columns.append(np.tile(gaspari_cohn(distance, length), n_blocks))
    return np.column_stack(columns)


# ---------------------------------------------------------------------- ES-MDA helpers
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

    There is no localisation: the distance localisation of the EnKF (applied to the state-data covariance only) made this
    update overshoot when the data of all dates are stacked (see ``docs/history-matching.md``)."""
    return enkf_analysis(Y, D, d_obs, np.asarray(sigma) * np.sqrt(alpha), rng)


# ---------------------------------------------------------------------- the prior


def check_prior(prior, params):
    """Raise ``ValueError`` unless ``prior`` (``{parameter name: entry}``) fits the parameter set: one entry per
    parameter, with only the known keys and a positive ``std`` (the JSON reader in ``spec.py`` adds located messages)."""
    names = [p.name for p in params.parameters]
    if set(prior) != set(names):
        raise ValueError(f"the prior needs exactly one entry per parameter {names}, got {sorted(prior)}")
    for name, entry in prior.items():
        unknown = set(entry) - _PRIOR_ENTRY_KEYS
        if unknown:
            raise ValueError(f"prior of {name}: unknown key(s) {sorted(unknown)}")
        if not entry.get("std", 0) > 0:
            raise ValueError(f"prior of {name}: std (in the transformed space, ln units for a log parameter) must be positive")
        if ("mean" in entry) and ("mean_by_layer" in entry):
            raise ValueError(f"prior of {name}: give mean or mean_by_layer, not both")


def _prior_mean(model, p, entry):
    """The prior mean of a parameter in the transformed space (a vector of the length of the parameter)."""
    grid = model.grid
    if isinstance(p, FieldParameter):
        if "mean_by_layer" in entry:
            by_layer = np.asarray(entry["mean_by_layer"], dtype=float)
            if by_layer.shape != (grid.nz,):
                raise ValueError(f"prior of {p.name}: mean_by_layer needs {grid.nz} values (one per layer)")
            physical = by_layer[grid.k]
        elif "mean" in entry:
            physical = np.broadcast_to(np.asarray(entry["mean"], dtype=float), (grid.n_cells,))
        else:
            physical = np.asarray(model.get(p.field), dtype=float)
    else:
        physical = np.atleast_1d(np.asarray(entry.get("mean", model.get(p.path)), dtype=float))
    return _forward(p.transform, np.asarray(physical, dtype=float), p.lower, p.upper)


def _member_streams(seed, members):
    analysis, *per_member = np.random.SeedSequence(seed).spawn(1 + members)
    return analysis, per_member


def sample_prior(model, params, prior, members, seed=11):
    """``members`` models drawn from the prior (see the module docstring); returns ``(models, matrix)`` where ``matrix``
    (size x members) holds the parameter vectors in the transformed space."""
    check_prior(prior, params)
    if members < 2:
        raise ValueError("an ensemble needs at least 2 members")
    layout = params.layout(model)
    means, chols = {}, {}
    for p in params.parameters:
        entry = prior[p.name]
        means[p.name] = _prior_mean(model, p, entry)
        if isinstance(p, FieldParameter) and ("range_xy" in entry or "range_z" in entry):
            if model.grid.n_cells > MAX_CORRELATED_CELLS:
                raise ValueError(f"prior of {p.name}: a correlated field needs a dense {model.grid.n_cells} x "
                                 f"{model.grid.n_cells} covariance; the limit is {MAX_CORRELATED_CELLS} blocks "
                                 "(leave out range_xy and range_z for independent blocks)")
            range_xy = entry.get("range_xy", entry.get("range_z"))
            range_z = entry.get("range_z", range_xy)
            chols[p.name] = np.linalg.cholesky(log_field_covariance(model.grid, entry["std"], range_xy, range_z))
    _, streams = _member_streams(seed, members)
    columns = []
    for stream in streams:
        rng = np.random.default_rng(stream)
        y = np.empty(params.size(model))
        for p in params.parameters:
            noise = rng.standard_normal(means[p.name].size)
            y[layout[p.name]] = means[p.name] + (chols[p.name] @ noise if p.name in chols else prior[p.name]["std"] * noise)
        columns.append(y)
    matrix = np.column_stack(columns)
    models = [params.apply(model, matrix[:, j], note=f"prior member {j}") for j in range(members)]
    return models, params.matrix(models)


# ---------------------------------------------------------------------- the two methods
def _member_rms(observations, records):
    return float(np.mean([rms(misfit(observations, r.table)) for r in records]))


def _metrics(params, observations, prior_models, models, prior_records, records, **extra):
    prior_x, post_x = params.matrix(prior_models), params.matrix(models)
    tables = [misfit(observations, r.table) for r in records]
    by_vector = {v: float(np.mean([np.sqrt(np.mean(t[t.vector == v].normalised.to_numpy() ** 2)) for t in tables]))
                 for v in observations.vectors}
    return dict(
        members=len(models), prior_member_rms=_member_rms(observations, prior_records),
        posterior_member_rms=_member_rms(observations, records), posterior_rms_by_vector=by_vector,
        prior_spread=float(prior_x.std(axis=1, ddof=1).mean()), posterior_spread=float(post_x.std(axis=1, ddof=1).mean()),
        mean_change=float(np.sqrt(np.mean((post_x.mean(axis=1) - prior_x.mean(axis=1)) ** 2))),
        posterior_ok=sum(r.ok for r in records), prior_ok=sum(r.ok for r in prior_records), **extra)


def _finish(model, params, observations, prior_models, models, prior_records, records, schedule, **extra):
    metrics = _metrics(params, observations, prior_models, models, prior_records, records, **extra)
    mean_vector = params.matrix(models).mean(axis=1)
    mean_model = params.apply(model, mean_vector, note="posterior ensemble mean")
    mean_record = run_to(mean_model, observations.times, schedule, observations.vectors)
    metrics["posterior_mean_rms"] = float(rms(misfit(observations, mean_record.table)))
    metrics["posterior_mean_ok"] = bool(mean_record.ok)
    member_rms = [rms(misfit(observations, r.table)) for r in records]
    metrics["best_member"] = int(np.argmin(member_rms))
    return dict(prior_models=prior_models, posterior_models=models, posterior_mean_model=mean_model, params=params,
                prior_records=prior_records, posterior_records=records, metrics=metrics)


def _pool(workers):
    return ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()


def run_esmda(model, params: ParameterSet, prior: dict, observations, members=40, seed=11, alphas: Optional[Sequence[float]] = None,
              n_passes=4, schedule=None, workers=1):
    """ES-MDA (see the module docstring). Returns a dictionary: ``prior_models``, ``posterior_models``,
    ``posterior_mean_model``, ``prior_records``, ``posterior_records`` and ``metrics`` (``rms_by_pass``: the data misfit,
    RMS of residual / sigma averaged over the members, before the first pass and after each pass; ``posterior_member_rms``
    is the last of them).

    Every observation time must be a multiple of ``dt``; ``schedule`` applies to all members."""
    alphas = list(alphas) if alphas is not None else esmda_alphas(n_passes)
    check_alphas(alphas)
    dates, vectors = observations.times, observations.vectors
    prior_models, _ = sample_prior(model, params, prior, members, seed)
    analysis_rng = np.random.default_rng(_member_streams(seed, members)[0])
    with _pool(workers) as pool:
        run = lambda models: run_ensemble(models, dates, schedule=schedule, vectors=vectors, executor=pool)
        prior_records = run(prior_models)
        models, records = list(prior_models), prior_records
        rms_by_pass, not_ok_by_pass = [_member_rms(observations, records)], [sum(not r.ok for r in records)]
        for number, alpha in enumerate(alphas, 1):
            d_obs, sigma, D = stack_data(observations, dates, records)
            Ya = esmda_update(params.matrix(models), D, d_obs, sigma, alpha, analysis_rng)
            models = [params.apply(m, Ya[:, j], note=f"ES-MDA pass {number} of {len(alphas)} (alpha {alpha:g})")
                      for j, m in enumerate(models)]
            records = run(models)                                  # the replay of this pass, also the next forecast
            rms_by_pass.append(_member_rms(observations, records))
            not_ok_by_pass.append(sum(not r.ok for r in records))
    return _finish(model, params, observations, prior_models, models, prior_records, records, schedule,
                   method="esmda", alphas=alphas, rms_by_pass=rms_by_pass, not_ok_by_pass=not_ok_by_pass)


def run_enkf(model, params: ParameterSet, prior: dict, observations, members=40, seed=11, schedule=None, workers=1,
             localise_length: Optional[float] = None):
    """EnKF (see the module docstring), date by date through the observation times. Returns the same dictionary as
    ``run_esmda`` (the posterior records are the replay from day 0); ``metrics`` has ``saturations_clipped``, the number
    of saturations that an update pushed outside their limits and that were put back.

    ``localise_length`` (ft) switches on Gaspari-Cohn localisation of the state-data covariance around the wells; it needs
    a parameter set of fields only."""
    dates, vectors = observations.times, observations.vectors
    if localise_length is not None and not all(isinstance(p, FieldParameter) for p in params.parameters):
        raise ValueError("localisation needs a parameter set of fields (one value per block) only")
    prior_models, _ = sample_prior(model, params, prior, members, seed)
    analysis_rng = np.random.default_rng(_member_streams(seed, members)[0])
    n = model.grid.n_cells
    n_par = params.size(model)
    n_blocks = n_par // n + 2 if localise_length is not None else None
    with _pool(workers) as pool:
        run = lambda ms, times, states=None: run_ensemble(ms, times, schedule=schedule, vectors=vectors, states=states,
                                                          executor=pool)
        prior_records = run(prior_models, dates)
        models, states, clipped = list(prior_models), [None] * members, 0
        for date in dates:
            records = run(models, [date], states)                                                       # forecast
            clipped += sum(r.adjusted.get("n_saturation_clipped", 0) for r in records)
            names, d_obs, sigma = observations.at(date)
            D = data_matrix(records, date, names)
            Y = np.vstack([params.matrix(models),
                           np.column_stack([r.state.p_o for r in records]),
                           np.column_stack([r.state.s_w for r in records])])
            rho_yd = None if localise_length is None else localisation_yd(model, names, localise_length, n_blocks)
            Ya = enkf_analysis(Y, D, d_obs, sigma, analysis_rng, rho_yd=rho_yd)                         # analysis
            models = [params.apply(m, Ya[:n_par, j], note=f"EnKF update at day {date:g}") for j, m in enumerate(models)]
            states = [dataclasses.replace(r.state, p_o=Ya[n_par:n_par + n, j], s_w=Ya[n_par + n:, j])
                      for j, r in enumerate(records)]
        records = run(models, dates)                                                                    # consistency replay
    return _finish(model, params, observations, prior_models, models, prior_records, records, schedule,
                   method="enkf", saturations_clipped=int(clipped), localise_length=localise_length)
