"""A twin experiment for ES-MDA on the kusim API (used by ``07_esmda_twin_experiment.py``).

The truth, its noisy data and the prior ensemble are those of ``enkf_twin.make_twin``, so the result can be compared with
the Ensemble Kalman Filter (``04``) and the adjoint match (``05``) on the same problem.  Each pass is

    run every member over the whole history from day 0
      -> compare with all the data at once and update the log permeability (``esmda.esmda_update``)
      -> edit every member with the updated log kx (``ParameterSet.apply``)

and after the last pass the members are run once more, to see how well the updated *models* reproduce the history.
There are no restarts and no update of pressure or saturation: the models are always consistent with their parameters.
"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Optional, Sequence

import numpy as np

from enkf_twin import VECTORS, make_twin
from esmda import check_alphas, esmda_alphas, esmda_update, stack_data
from kusim.history import misfit, rms, run_ensemble
from kusim.parameters import FieldParameter, ParameterSet


def run_esmda_experiment(ne=40, seed=11, n_passes=4, alphas: Optional[Sequence[float]] = None, dt=10.0, t_end=600.0,
                         first_date=50.0, interval=50.0, std=0.7, workers=1):
    """Run the experiment and return everything in a dictionary (models, records, observations, metrics).

    ``alphas`` are the inflation factors of the passes (default: ``n_passes`` equal ones, ``alpha = n_passes``).
    The set-up (truth, data, prior) is that of ``enkf_twin.make_twin``.  ``metrics`` has the same keys as that of the EnKF
    twin (except the saturation clipping, which does not occur here), and ``rms_by_pass``: the data misfit (RMS of the
    residual / sigma, mean over members) before the first pass and after each pass; the last entry is ``posterior_member_rms``."""
    alphas = list(alphas) if alphas is not None else esmda_alphas(n_passes)
    check_alphas(alphas)
    twin = make_twin(ne, seed, dt, t_end, first_date, interval, std)
    dates, grid, truth_model, truth = twin["dates"], twin["grid"], twin["truth_model"], twin["truth"]
    obs, prior_models = twin["observations"], twin["prior_models"]
    params = ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=1e4, ties=("ky", "kz"))])
    analysis_rng = np.random.default_rng(twin["analysis_seed"])
    member_rms = lambda recs: float(np.mean([rms(misfit(obs, r.table)) for r in recs]))

    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        run = lambda models: run_ensemble(models, dates, vectors=VECTORS, executor=pool)
        prior_records = run(prior_models)
        models, records = list(prior_models), prior_records
        rms_by_pass, not_ok_by_pass = [member_rms(records)], [sum(not r.ok for r in records)]
        for number, alpha in enumerate(alphas, 1):
            d_obs, sigma, D = stack_data(obs, dates, records)
            Ya = esmda_update(params.matrix(models), D, d_obs, sigma, alpha, analysis_rng)
            models = [params.apply(m, Ya[:, j], note=f"ES-MDA pass {number} of {len(alphas)} (alpha {alpha:g})")
                      for j, m in enumerate(models)]
            records = run(models)                                      # the replay of this pass, also the next forecast
            rms_by_pass.append(member_rms(records))
            not_ok_by_pass.append(sum(not r.ok for r in records))
    posterior_records = records

    truth_logk = np.log(truth_model.kx)
    prior_x, post_x = params.matrix(prior_models), params.matrix(models)
    metrics = dict(
        prior_member_rms=rms_by_pass[0], posterior_member_rms=rms_by_pass[-1], rms_by_pass=rms_by_pass,
        not_ok_by_pass=not_ok_by_pass, alphas=alphas,
        prior_logk_rmse=float(np.sqrt(np.mean((prior_x.mean(axis=1) - truth_logk) ** 2))),
        posterior_logk_rmse=float(np.sqrt(np.mean((post_x.mean(axis=1) - truth_logk) ** 2))),
        prior_spread=float(prior_x.std(axis=1, ddof=1).mean()), posterior_spread=float(post_x.std(axis=1, ddof=1).mean()),
        posterior_ok=sum(r.ok for r in posterior_records), members=ne)
    return dict(dates=dates, grid=grid, truth_model=truth_model, truth=truth, observations=obs, params=params,
                prior_models=prior_models, posterior_models=models, prior_records=prior_records,
                posterior_records=posterior_records, metrics=metrics)
