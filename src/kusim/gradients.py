"""Objective and gradient of a history-matching or control problem, by adjoint and by finite differences.

A ``HistoryProblem`` ties together a model, observed data and the things to vary, and treats them as one vector ``x``:

    problem = HistoryProblem(model, observations, params=ParameterSet([...]),            # e.g. ln permeability, porosity
                             controls=ControlSet([...]), initial_controls={"I1": [9000, 9000, 9000]},
                             base_schedule=history)
    x0 = problem.vector0()
    J = problem.value(x0)                          # runs the simulator and compares with the data
    J, g, result = problem.value_and_gradient(x0)  # the same objective and its gradient by the adjoint (kusim.adjoint)
    g_fd = problem.fd_gradient(x0, workers=4)      # finite differences, for checking (one or two runs per component)
    problem.check(x0)                              # adjoint against finite differences for some components

``x`` is the parameter vector (``ParameterSet``, transformed space: ln k, ...) followed by the control vector
(``ControlSet``, one target per control period).  The objective is ``J = sum weight * ((simulated - observed) / sigma)**2``
(Intersect eq. 2.1045).  ``value`` goes through the ``Simulator`` and ``value_and_gradient`` through the functional copy
of ``kusim.adjoint``, so a comparison of the two also checks that copy against the simulator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .adjoint import adjoint_gradient
from .history import evaluate_linear_terms, misfit, objective, run_to


def fd_gradient(f, x, h=1e-4, central=True, executor=None, indices=None):
    """Finite-difference gradient of the scalar function ``f`` at ``x``.

    The step of component i is ``h * max(1, |x_i|)``.  ``central=True`` uses two evaluations per component (error of
    order h**2), otherwise one.  ``indices`` limits the components (the others are NaN).  ``executor`` (a
    ``concurrent.futures`` executor, ``f`` must be picklable) evaluates the runs in parallel."""
    x = np.asarray(x, dtype=float)
    indices = range(x.size) if indices is None else list(indices)
    steps = {i: h * max(1.0, abs(x[i])) for i in indices}
    points = []
    for i in indices:
        xp = x.copy()
        xp[i] += steps[i]
        points.append(xp)
        if central:
            xm = x.copy()
            xm[i] -= steps[i]
            points.append(xm)
    if not central:
        points.append(x.copy())
    values = list(executor.map(f, points)) if executor is not None else [f(pt) for pt in points]
    grad = np.full(x.size, np.nan)
    for j, i in enumerate(indices):
        if central:
            grad[i] = (values[2 * j] - values[2 * j + 1]) / (2 * steps[i])
        else:
            grad[i] = (values[j] - values[-1]) / steps[i]
    return grad


class HistoryProblem:
    """A model, observations and the variables to change (see the module docstring)."""

    def __init__(self, model, observations=None, params=None, controls=None, base_schedule=None, state=None,
                 initial_controls: Optional[dict] = None, upstream: str = "potential", linear_terms=None, prior=None):
        if params is None and controls is None:
            raise ValueError("give params and/or controls: there is nothing to vary")
        if observations is None and linear_terms is None:
            raise ValueError("give observations and/or linear_terms: there is no objective")
        if controls is not None and initial_controls is None:
            raise ValueError("controls need initial_controls={well: [target of period 0, 1, ...]}")
        self.model, self.observations, self.params, self.controls = model, observations, params, controls
        self.base_schedule, self.state, self.upstream, self.initial_controls = base_schedule, state, upstream, initial_controls
        self.linear_terms = linear_terms
        self.n_params = 0 if params is None else params.size(model)
        self.n_controls = 0 if controls is None else controls.size
        self.prior = None
        if prior is not None:
            if params is None:
                raise ValueError("a prior is on the parameters: give params")
            mean, std = (np.broadcast_to(np.asarray(a, dtype=float), (self.n_params,)).copy() for a in prior)
            if np.any(std <= 0):
                raise ValueError("every prior standard deviation must be positive")
            self.prior = (mean, std)

    # ------------------------------------------------------------------
    def vector0(self):
        """The starting vector: the parameters of the model, then the initial control targets."""
        parts = []
        if self.params is not None:
            parts.append(self.params.vector(self.model))
        if self.controls is not None:
            parts.append(self.controls.vector(self.initial_controls))
        return np.concatenate(parts)

    def bounds(self):
        """``(lower, upper)`` for an optimiser (parameters in the transformed space, then the controls)."""
        lower, upper = [], []
        if self.params is not None:
            lo, hi = self.params.bounds(self.model)
            lower.append(lo)
            upper.append(hi)
        if self.controls is not None:
            lo, hi = self.controls.bounds()
            lower.append(lo)
            upper.append(hi)
        return np.concatenate(lower), np.concatenate(upper)

    def split(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_params + self.n_controls,):
            raise ValueError(f"x needs {self.n_params + self.n_controls} values, got {x.size}")
        return x[: self.n_params], x[self.n_params:]

    def build(self, x):
        """The model and the schedule that the vector ``x`` stands for."""
        y, u = self.split(x)
        model = self.params.apply(self.model, y) if self.params is not None else self.model
        schedule = self.controls.schedule(u, self.base_schedule) if self.controls is not None else self.base_schedule
        return model, schedule

    # ------------------------------------------------------------------
    def _prior_value_and_gradient(self, x):
        """The prior term ``sum ((y - mean) / std)**2`` of the parameters (in the transformed space) and its gradient."""
        if self.prior is None:
            return 0.0, np.zeros(self.n_params + self.n_controls)
        mean, std = self.prior
        r = (self.split(x)[0] - mean) / std
        return float(np.sum(r**2)), np.concatenate([2.0 * r / std, np.zeros(self.n_controls)])

    def value(self, x):
        """The objective through the simulator (an independent path from the adjoint).

        ``sum weight * misfit**2`` over the observations, plus the linear terms, plus the prior term."""
        model, schedule = self.build(x)
        times, vectors = set(), set()
        for source in (self.observations.table if self.observations is not None else None, self.linear_terms):
            if source is not None:
                times |= set(source.time)
                vectors |= set(source.vector)
        table = run_to(model, sorted(times), schedule, sorted(vectors), self.state, self.upstream).table
        total = self._prior_value_and_gradient(x)[0]
        if self.observations is not None:
            total += objective(misfit(self.observations, table))
        if self.linear_terms is not None:
            total += evaluate_linear_terms(self.linear_terms, table)
        return total

    def value_and_gradient(self, x):
        """``(J, dJ/dx, AdjointResult)``: the objective and its gradient by the adjoint (about the cost of 2-3 runs)."""
        y, u = self.split(x)
        model, schedule = self.build(x)
        result = adjoint_gradient(model, self.observations, schedule, self.state, self.upstream, self.linear_terms)
        parts = []
        if self.params is not None:
            parts.append(self.params.pullback(self.model, y, result.fields))
        if self.controls is not None:
            parts.append(self.controls.pullback(u, result.controls, result.step_times))
        prior_value, prior_gradient = self._prior_value_and_gradient(x)
        return result.value + prior_value, np.concatenate(parts) + prior_gradient, result

    def fd_gradient(self, x, h=1e-4, central=True, executor=None, indices=None):
        """The gradient by finite differences of ``value`` (see ``fd_gradient``)."""
        return fd_gradient(self.value, x, h, central, executor, indices)

    def check(self, x, indices=None, h=(1e-4, 1e-5, 1e-6, 1e-7), n=8, seed=0, stable_tol=1e-5):
        """The adjoint gradient against finite differences, for some components of ``x``.

        Returns a table with one row per component: the ``adjoint`` value, the ``finite_difference`` value, ``relative`` (their
        difference in units of the finite-difference value) and ``stable``.  Finite differences have two errors, truncation
        (large step: the curvature of the objective) and round-off (small step), and the objective of an upwind scheme also
        has kinks (a switch of the upstream block, a clipped saturation, a table node), so one step size proves little.
        Every step in ``h`` (relative: the step is ``h * max(1, |x_i|)``) is tried; the estimate counts as **stable** when two
        neighbouring steps agree to ``stable_tol``, and then ``finite_difference`` is the one of the smaller step (less
        truncation error).  Components for which no pair
        agrees are not stable: ``relative`` is NaN for them, they are probably at a kink (see ``result.diagnostics``).
        Compare the adjoint with the stable estimates only.  ``indices`` default to ``n`` random components.

        The smallest steps matter: the objective of this scheme has jumps and kinks at scales of a fraction of a STB/day
        in a rate (see ``docs/adjoint.md``), so the estimates at large steps can be off by percent while those at 1e-6 and
        1e-7 agree with the adjoint to many digits."""
        x = np.asarray(x, dtype=float)
        _, g, result = self.value_and_gradient(x)
        if indices is None:
            indices = sorted(np.random.default_rng(seed).choice(x.size, size=min(n, x.size), replace=False))
        indices = [int(i) for i in indices]
        steps = sorted(np.atleast_1d(h).astype(float), reverse=True)
        estimates = np.array([self.fd_gradient(x, h=step, indices=indices)[indices] for step in steps])   # (steps, components)
        scale = max(np.max(np.abs(g[indices])), 1e-300) if indices else 1.0
        floor = 1e-3 * scale
        rows = []
        for j, i in enumerate(indices):
            fd, stable = np.nan, False
            for a, b in zip(estimates[:-1, j], estimates[1:, j]):
                if abs(a - b) <= stable_tol * max(abs(a), abs(b), floor):
                    fd, stable = b, True
                    break
            rows.append(dict(index=i, adjoint=g[i], finite_difference=fd, stable=stable,
                             relative=abs(g[i] - fd) / max(abs(fd), floor) if stable else np.nan))
        return pd.DataFrame(rows)
