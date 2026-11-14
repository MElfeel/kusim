"""Two workflows with adjoint gradients on the kusim API (used by ``05_adjoint_history_matching.py`` and
``06_adjoint_production_optimisation.py``): history matching and production optimisation of the twin model of ``enkf_twin.py``.

Both use ``kusim.gradients.HistoryProblem`` (objective and adjoint gradient) and SciPy's L-BFGS-B.  The results are always
checked by running the simulator again, because the objective of an IMPES scheme has jumps.
"""

import time

import numpy as np
from scipy.optimize import minimize

from enkf_reference import log_field_covariance
from enkf_twin import LAYER_MEAN_KX, VECTORS, make_twin
from kusim.gradients import HistoryProblem
from kusim.history import linear_terms, misfit, rms, run_to
from kusim.parameters import FieldParameter, ParameterSet
from kusim.schedule import ControlParameter, ControlSet


# ====================================================================== history matching
def history_match(seed=0, maxiter=150, std=0.7, correlated_prior=True, time_finite_differences=False):
    """Maximum a posteriori estimate of the log-permeability of the twin from its noisy well data.

    The prior is Gaussian with the mean and covariance that generated the twin (an exponential correlation of 1200 ft
    and 60 ft).  It is used as a *reparameterisation*: the unknowns are ``z`` with the prior N(0, I), and the log
    permeability is ``mean + L z`` with ``L`` the Cholesky factor of the covariance, so that the objective is
    ``data misfit + z.z`` and its gradient is ``L^T (adjoint gradient) + 2 z``.  With ``correlated_prior=False`` the blocks
    have independent priors of standard deviation ``std`` instead (which ignores the spatial correlation).
    """
    twin = make_twin(ne=1, seed=seed, std=std)
    grid, obs, dates = twin["grid"], twin["observations"], twin["dates"]
    base = twin["prior_models"][0]                                    # only its structure is used: kx is replaced
    params = ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=1e4, ties=("ky", "kz"))])
    mean = np.log(np.asarray(LAYER_MEAN_KX))[grid.k]
    chol = np.linalg.cholesky(log_field_covariance(grid, std, 1200.0, 60.0)) if correlated_prior else std * np.eye(grid.n_cells)
    problem = HistoryProblem(base, obs, params=params)                # the data misfit; the prior term is added below
    history = []

    def fun(z):
        J, g, _ = problem.value_and_gradient(mean + chol @ z)
        history.append(J)                                              # the data part of the objective
        return J + z @ z, chol.T @ g + 2.0 * z

    z0 = np.zeros(grid.n_cells)
    start = time.perf_counter()
    result = minimize(fun, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=2 * maxiter))
    seconds = time.perf_counter() - start
    x = mean + chol @ result.x

    one_run = _time(lambda: problem.value(mean))
    one_gradient = _time(lambda: problem.value_and_gradient(mean))
    out = dict(twin=twin, params=params, problem=problem, mean=mean, x=x, z=result.x, history=history, iterations=result.nit,
               evaluations=result.nfev, seconds=seconds, message=result.message, seconds_per_run=one_run,
               seconds_per_gradient=one_gradient, check=problem.check(mean, n=8))
    if time_finite_differences:
        out["seconds_per_fd_gradient"] = _time(lambda: problem.fd_gradient(mean, h=1e-6), repeat=1)

    truth_logk = np.log(twin["truth_model"].kx)

    def table(vec):
        return run_to(params.apply(base, vec), dates, vectors=VECTORS).table

    out["start_table"], out["end_table"] = table(mean), table(x)
    out["metrics"] = dict(
        start_data_rms=rms(misfit(obs, out["start_table"])), end_data_rms=rms(misfit(obs, out["end_table"])),
        start_logk_rmse=float(np.sqrt(np.mean((mean - truth_logk) ** 2))), end_logk_rmse=float(np.sqrt(np.mean((x - truth_logk) ** 2))),
        start_layers=mean.reshape(3, -1).mean(axis=1), end_layers=x.reshape(3, -1).mean(axis=1),
        truth_layers=truth_logk.reshape(3, -1).mean(axis=1))
    return out


def _time(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


# ====================================================================== production optimisation
PERIODS = [0.0, 150.0, 300.0, 450.0]


def allocate_injection(prices=None, total=9000.0, lower=2000.0, upper=14000.0, dt=10.0, seed=0, compare_nelder_mead=False):
    """Distribute a fixed volume of injection water over four periods of 150 days to maximise the profit at 600 days.

    The producer is on bottom-hole-pressure control (2500 psia) and the injector on rate control, with a rate in each period
    between ``lower`` and ``upper`` STB/day and the average rate fixed at ``total``: the volume is the constraint, which
    SLSQP handles with the gradient of the adjoint.  The profit is the sum of the prices times the cumulative volumes at 600
    days: ``prices`` maps a vector (``FOPT``, ``FWPT``, ``FWIT``) to its value per STB; the default is 60 for oil, minus 20 for
    produced water and minus 3 for injected water.

    The wells are chosen so that the result is physical.  A producer on a rate limit (``min_bhp``) switches to pressure control
    during the run, which makes the objective jump, and an optimiser sits on that jump; a bottom-hole pressure beyond the
    pressure of the block makes a well flow backwards, which an optimiser exploits.
    The run is checked for them along the whole path of the optimiser (``backflow_on_the_path``, ``mode_switches_on_the_path``).
    """
    prices = prices or {"FOPT": 60.0, "FWPT": -20.0, "FWIT": -3.0}
    model = make_twin(ne=1, seed=seed)["truth_model"].with_changes({
        "dt": dt, "producers[0].mode": "bhp", "producers[0].target": 2500.0,
        "injectors[0].mode": "water_rate", "injectors[0].target": total})
    controls = ControlSet([ControlParameter("I1", PERIODS, mode="water_rate", lower=lower, upper=upper)])
    minus_profit = linear_terms(600.0, {v: -price for v, price in prices.items()})        # minimise J = - profit
    problem = HistoryProblem(model, controls=controls, initial_controls={"I1": [total] * 4}, linear_terms=minus_profit)
    x0 = problem.vector0()
    scale = 1e-3                                                      # the variables are thousands of STB/day, the profit millions
    history, worst = [], dict(backflow=0, switches=0)

    def fun(u):
        J, g, result = problem.value_and_gradient(u / scale)
        history.append(-J)
        worst["backflow"] = max(worst["backflow"], result.diagnostics["backflow_well_steps"])
        worst["switches"] = max(worst["switches"], result.diagnostics["mode_switches"])
        return J * 1e-6, g * 1e-6 / scale

    volume = {"type": "eq", "fun": lambda u: u.sum() - len(u) * total * scale, "jac": lambda u: np.ones_like(u)}
    start = time.perf_counter()
    result = minimize(fun, x0 * scale, jac=True, method="SLSQP", bounds=[(lower * scale, upper * scale)] * len(x0),
                      constraints=[volume], options=dict(maxiter=100, ftol=1e-10))
    seconds = time.perf_counter() - start
    x = result.x / scale
    out = dict(problem=problem, x0=x0, x=x, history=history, iterations=result.nit, evaluations=result.nfev, seconds=seconds,
               message=result.message, profit_start=-problem.value(x0), profit_end=-problem.value(x),     # by the simulator
               backflow_on_the_path=worst["backflow"], mode_switches_on_the_path=worst["switches"],
               check=problem.check(x0), seconds_per_gradient=_time(lambda: problem.value_and_gradient(x0)),
               seconds_per_run=_time(lambda: problem.value(x0)))
    final_model, final_schedule = problem.build(x)
    out["final"] = run_to(final_model, [600.0], final_schedule, vectors=["FOPT", "FWPT", "FWIT"])
    if compare_nelder_mead:                                           # three free rates; the fourth follows from the volume
        def reduced(v):
            u = np.append(v, len(x0) * total * scale - v.sum())
            return 1e3 if not (lower * scale <= u[-1] <= upper * scale) else problem.value(u / scale) * 1e-6
        nm = minimize(reduced, x0[:-1] * scale, method="Nelder-Mead", bounds=[(lower * scale, upper * scale)] * (len(x0) - 1),
                      options=dict(maxfev=400, xatol=1e-3, fatol=1e-9))
        u = np.append(nm.x, len(x0) * total * scale - nm.x.sum()) / scale
        out["nelder_mead"] = dict(profit=-problem.value(u), evaluations=nm.nfev, x=u)
    return out
