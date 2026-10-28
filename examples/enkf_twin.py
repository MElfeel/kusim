"""A twin experiment for an Ensemble Kalman Filter on the kusim API (used by ``04_enkf_twin_experiment.py``).

A "truth" model with a known, correlated log-permeability field produces noisy data (well pressures and water
cut).  A prior ensemble with the same statistics but other realisations is then history matched with the filter
of ``enkf_reference.py``, assimilating the data date by date:

    forecast the members to the next date (each restarts from its own state)
      -> analysis of the augmented state [log kx, pressure, Sw] with the data
      -> edit every member with the updated log kx (``ParameterSet.apply``) and restart it from the updated
         pressure and saturation (``SimState`` + ``Simulator.set_state``, which clips Sw to its limits)

Afterwards every updated member is run again from time zero: the static and the dynamic variables of an EnKF update
are not consistent (the filter changed them separately), and this replay is the check that the updated *models*
reproduce the history (the "rerun from time zero" of ``enkf/HA_localised_enkf.docx``).

The twin is a test of the workflow, not of the filter: with two wells and three data types the data say little about
72 permeabilities, so the field is recovered only partly while the data are matched well.
"""

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Optional

import numpy as np

from enkf_reference import correlated_log_field, enkf_analysis
from kusim import Equilibrium, Grid, Injector, Producer, PVTTable, ReservoirModel, RockFluidTable
from kusim.ensemble import localisation_yd  # noqa: F401  (re-exported)
from kusim.history import Observations, data_matrix, misfit, rms, run_ensemble, run_to
from kusim.parameters import FieldParameter, ParameterSet

LAYER_MEAN_KX = [120.0, 600.0, 250.0]                     # md, the bonus-project layers
VECTORS = ["WBHP:P1", "WBHP:I1", "WWCT:P1"]
SIGMA = {"WBHP:P1": 10.0, "WBHP:I1": 10.0, "WWCT:P1": 0.02}


def make_model(kx, dt=10.0, t_end=600.0):
    """The 6 x 4 x 3 layered waterflood of the bonus project with the permeability ``kx`` (one value per block)."""
    grid = Grid([400.0] * 6, [400.0] * 4, [40.0] * 3, top_depth=7500.0)
    rock_fluid = RockFluidTable([.2, .4, .6, .75, .85, 1.], [.85, .35, .1, .02, 0, 0], [0, .05, .2, .4, .6, 1.],
                                [8, 2.5, .8, .3, .15, 0])
    pvt = PVTTable([400., 1000., 5500.], [1.30, 1.24, 1.17], [1.01] * 3, [.9, 1.0, 1.8], [.7] * 3)
    gamma_o, gamma_w = 50 / 62.4, 64 / 62.4
    return ReservoirModel(
        grid, kx, 0.8 * kx, 0.1 * kx, 0.22, 5e-6, pvt, rock_fluid, gamma_o, gamma_w, 0.2, 0.15,
        Equilibrium(4200.0 - 0.433 * gamma_o * 7580.0, 8000.0), dt, t_end,
        producers=[Producer(grid.column_blocks(1, 1), "liquid_rate", 8000.0, radius=0.35, min_bhp=1500.0)],
        injectors=[Injector(grid.column_blocks(6, 4), "water_rate", 9000.0, radius=0.35)])


def make_twin(ne=40, seed=11, dt=10.0, t_end=600.0, first_date=50.0, interval=50.0, std=0.7):
    """The truth, its noisy data and ``ne`` prior models: the set-up shared by the EnKF and the adjoint examples.

    The random numbers come from independent streams of ``seed``: the truth, the measurement noise and the filter's
    perturbations do not depend on ``ne``, and member j of a larger ensemble is member j of a smaller one.
    Raises ValueError if the random truth leaves the PVT table (use another seed): such data would not be a fair test."""
    dates = [float(t) for t in np.arange(first_date, t_end + 1e-9, interval)]
    truth_seed, noise_seed, analysis_seed, *member_seeds = np.random.SeedSequence(seed).spawn(3 + ne)
    grid = make_model(100.0, dt, t_end).grid

    def field(stream):
        return correlated_log_field(grid, LAYER_MEAN_KX, std, 1200.0, 60.0, np.random.default_rng(stream))[:, 0]

    truth_model = make_model(np.exp(field(truth_seed)), dt, t_end)
    prior_models = [make_model(np.exp(field(s)), dt, t_end) for s in member_seeds]
    truth = run_to(truth_model, dates, vectors=VECTORS)
    if not truth.ok:
        raise ValueError(f"the truth for seed {seed} leaves the PVT table (pressure {truth.diagnostics['p_min']:.0f} to "
                         f"{truth.diagnostics['p_max']:.0f} psia): use another seed")
    obs = Observations.synthetic(truth.table, dates, SIGMA, np.random.default_rng(noise_seed))
    return dict(dates=dates, grid=grid, truth_model=truth_model, truth=truth, observations=obs, prior_models=prior_models,
                analysis_seed=analysis_seed)


def run_twin_experiment(ne=40, seed=11, dt=10.0, t_end=600.0, first_date=50.0, interval=50.0, std=0.7,
                        localise_length: Optional[float] = None, workers=1):
    """Run the experiment and return everything in a dictionary (models, records, observations, metrics).

    The set-up is that of ``make_twin`` (same arguments)."""
    twin = make_twin(ne, seed, dt, t_end, first_date, interval, std)
    dates, grid, truth_model, truth = twin["dates"], twin["grid"], twin["truth_model"], twin["truth"]
    obs, prior_models = twin["observations"], twin["prior_models"]
    params = ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=1e4, ties=("ky", "kz"))])
    analysis_rng = np.random.default_rng(twin["analysis_seed"])
    n = grid.n_cells

    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:       # one pool for the loop
        run = lambda models, times, states=None: run_ensemble(models, times, vectors=VECTORS, states=states, executor=pool)
        prior_records = run(prior_models, dates)
        models, states, clipped = list(prior_models), [None] * ne, 0
        for date in dates:
            records = run(models, [date], states)                                                       # forecast
            clipped += sum(r.adjusted.get("n_saturation_clipped", 0) for r in records)
            names, d_obs, sigma = obs.at(date)
            D = data_matrix(records, date, names)
            Y = np.vstack([params.matrix(models),
                           np.column_stack([r.state.p_o for r in records]),
                           np.column_stack([r.state.s_w for r in records])])
            rho_yd = None if localise_length is None else localisation_yd(truth_model, names, localise_length)
            Ya = enkf_analysis(Y, D, d_obs, sigma, analysis_rng, rho_yd=rho_yd)                         # analysis
            models = [params.apply(m, Ya[:n, j], note=f"EnKF update at day {date:g}") for j, m in enumerate(models)]
            states = [dataclasses.replace(r.state, p_o=Ya[n:2 * n, j], s_w=Ya[2 * n:, j]) for j, r in enumerate(records)]
        posterior_records = run(models, dates)                                                          # consistency replay

    truth_logk = np.log(truth_model.kx)
    prior_x, post_x = params.matrix(prior_models), params.matrix(models)
    member_rms = lambda recs: float(np.mean([rms(misfit(obs, r.table)) for r in recs]))
    metrics = dict(
        prior_member_rms=member_rms(prior_records), posterior_member_rms=member_rms(posterior_records),
        prior_logk_rmse=float(np.sqrt(np.mean((prior_x.mean(axis=1) - truth_logk) ** 2))),
        posterior_logk_rmse=float(np.sqrt(np.mean((post_x.mean(axis=1) - truth_logk) ** 2))),
        prior_spread=float(prior_x.std(axis=1, ddof=1).mean()), posterior_spread=float(post_x.std(axis=1, ddof=1).mean()),
        posterior_ok=sum(r.ok for r in posterior_records), members=ne, saturations_clipped=int(clipped))
    return dict(dates=dates, grid=grid, truth_model=truth_model, truth=truth, observations=obs, params=params,
                prior_models=prior_models, posterior_models=models, prior_records=prior_records,
                posterior_records=posterior_records, metrics=metrics)
