"""Tools for history matching and ensemble methods: observed data, running members, comparing with the data.

The pieces, in the order an Ensemble Kalman Filter uses them:

* ``Observations``: measured data (time, vector name, value, standard deviation of the error).
* ``run_ensemble``: run many models (members) to the next data date, each from its own restart state if given,
  optionally on several processes.  Every member returns a ``RunRecord``: the simulated vectors at the requested
  dates, the final ``SimState`` to continue from, and ``diagnostics``.
* ``data_matrix``: the simulated data of all members as one matrix (data x members), as the update equation wants it.
* ``misfit``: simulated against observed data, in units of the measurement error.

Vector names follow the usual summary vectors: ``WBHP:P1`` (bottom-hole pressure of well P1), ``WOPR:P1``,
``WWPR:P1``, ``WWCT:P1`` (water cut), ``WWIR:I1`` (water injection rate), ``FPRP`` (pore-volume average
pressure), ``FOPT``, ``FWPT``, ``FWIT`` (cumulative volumes).  Updating the model (permeabilities, ...) is the job
of ``kusim.parameters``; the analysis step of the filter itself is not part of the package.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .schedule import Schedule, advance
from .simulator import SimState, Simulator


# ---------------------------------------------------------------------- simulated data
def summary_vectors(sim) -> dict:
    """The summary vectors of the last step of ``sim`` (the rates are those of that step)."""
    out = {}
    for name, d in sim.well_data().items():
        out[f"WBHP:{name}"] = d["bhp"]
        if d["kind"] == "producer":
            liquid = d["oil_rate"] + d["water_rate"]
            out[f"WOPR:{name}"] = d["oil_rate"]
            out[f"WWPR:{name}"] = d["water_rate"]
            out[f"WWCT:{name}"] = d["water_rate"] / liquid if liquid > 0 else float("nan")
        else:
            out[f"WWIR:{name}"] = d["water_rate"]
    pore_volume = sim.grid.volume * sim.model.porosity
    out["FPRP"] = float((sim.p_o * pore_volume).sum() / pore_volume.sum())
    cum = sim.get_state().cum
    out["FOPT"] = sum(v.get("oil", 0.0) for k, v in cum.items() if k.startswith("P"))
    out["FWPT"] = sum(v["water"] for k, v in cum.items() if k.startswith("P"))
    out["FWIT"] = sum(v["water"] for k, v in cum.items() if k.startswith("I"))
    return out


@dataclass
class RunRecord:
    """What one member returns: ``table`` (a row of vectors per requested time, column TIME), the ``state`` at the
    last time to continue from, and the ``diagnostics`` of the simulator (see ``Simulator.diagnostics``).
    ``adjusted`` says what had to be changed in the restart state that the member was given (for example
    ``{"n_saturation_clipped": 3}`` when an update had pushed saturations outside their limits)."""

    table: pd.DataFrame
    state: SimState
    diagnostics: dict
    adjusted: dict = field(default_factory=dict)

    @property
    def ok(self):
        """No sign of trouble: pressure stayed inside the PVT table and no run stopped early."""
        d = self.diagnostics
        return not d["outside_pvt_range"] and d["stop_reason"] is None


def run_to(model, times: Sequence[float], schedule: Optional[Schedule] = None, vectors: Optional[Sequence[str]] = None,
           state: Optional[SimState] = None, upstream: str = "potential") -> RunRecord:
    """Run ``model`` through the increasing report ``times`` and record the summary vectors at each of them.

    The run starts at time 0, or from ``state`` (a restart: its time must not be after the first report time).
    ``schedule`` changes the well controls on the way.  All times must be multiples of the model's ``dt``.
    ``vectors`` (a list of names) restricts the recorded columns; the default records all.  ``upstream`` is the rule of
    ``Simulator`` for the upstream block.
    """
    sim = Simulator(model, save_every=10**9, upstream=upstream)             # the fields are not saved: only the state at the end
    adjusted = sim.set_state(state) if state is not None else {}
    rows = []
    for t in times:
        steps = advance(sim, t, schedule)
        if steps == 0 and sim.time < t - 1e-6:
            break                                         # a well stop condition ended the run
        if not sim.well_data():
            raise ValueError(f"report time {t} is not after the start of the run (day {sim.time})")
        data = summary_vectors(sim)
        rows.append({"TIME": sim.time, **(data if vectors is None else {v: data[v] for v in vectors})})
    return RunRecord(pd.DataFrame(rows), sim.get_state(), sim.diagnostics(), adjusted)


def run_ensemble(models, times, schedule=None, vectors=None, states=None, workers=1, executor=None) -> List[RunRecord]:
    """``run_to`` for every member.  ``states`` (a list, same length) restarts each member from its own state.

    With ``workers > 1`` the members run in separate processes (the models and states are copied to them).
    Starting the processes takes a while (seconds on Windows), so a loop that calls this many times should make one
    ``concurrent.futures.ProcessPoolExecutor`` and pass it as ``executor`` (``workers`` is then ignored).
    On Windows a script that starts processes must be guarded by ``if __name__ == "__main__":``.
    """
    models = list(models)
    states = [None] * len(models) if states is None else list(states)
    if len(states) != len(models):
        raise ValueError("states must have one entry per model")
    if executor is None and workers <= 1:
        return [run_to(m, times, schedule, vectors, s) for m, s in zip(models, states)]
    if executor is not None:
        return _submit_all(executor, models, states, times, schedule, vectors)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return _submit_all(pool, models, states, times, schedule, vectors)


def _submit_all(pool, models, states, times, schedule, vectors):
    futures = [pool.submit(run_to, m, times, schedule, vectors, s) for m, s in zip(models, states)]
    return [f.result() for f in futures]


def data_matrix(records, time, vectors) -> np.ndarray:
    """Simulated data of all members at one ``time`` as a matrix of shape (len(vectors), members)."""
    cols = []
    for r in records:
        row = r.table[np.isclose(r.table.TIME, time)]
        if row.empty:
            raise ValueError(f"a member has no data at time {time}")
        cols.append([float(row.iloc[0][v]) for v in vectors])
    return np.array(cols).T


# ---------------------------------------------------------------------- observed data
class Observations:
    """Measured data: a table with the columns ``time``, ``vector``, ``value`` and ``sigma``
    (the standard deviation of the measurement error, in the unit of the vector) and, optionally, ``weight``
    (a factor ``s`` of the objective, default 1: ``J = sum s * ((simulated - observed) / sigma)**2``)."""

    def __init__(self, table):
        table = pd.DataFrame(table).copy()
        missing = {"time", "vector", "value", "sigma"} - set(table.columns)
        if missing:
            raise ValueError(f"the observation table lacks the columns {sorted(missing)}")
        if (table.sigma <= 0).any():
            raise ValueError("every sigma must be positive")
        if "weight" not in table.columns:
            table["weight"] = 1.0
        if (table.weight < 0).any():
            raise ValueError("every weight must be zero or positive")
        columns = ["time", "vector", "value", "sigma", "weight"]
        self.table = table[columns].sort_values(["time", "vector"]).reset_index(drop=True)

    @property
    def times(self):
        return sorted(self.table.time.unique())

    @property
    def vectors(self):
        return sorted(self.table.vector.unique())

    def at(self, time):
        """``(vectors, values, sigmas)`` of the data measured at ``time`` (arrays in the same order)."""
        rows = self.table[np.isclose(self.table.time, time)]
        return list(rows.vector), rows.value.to_numpy(), rows.sigma.to_numpy()

    @classmethod
    def synthetic(cls, truth: pd.DataFrame, times, sigma: dict, rng) -> "Observations":
        """Noisy "measurements" from a run of a known model (a twin experiment).

        ``truth`` has the column TIME and the vectors; ``sigma`` maps vector name -> standard deviation; the
        measurement at each time is the true value plus ``sigma`` times a standard normal number from ``rng``."""
        rows = []
        for t in times:
            row = truth[np.isclose(truth.TIME, t)]
            if row.empty:
                raise ValueError(f"the truth has no data at time {t}")
            for v, s in sigma.items():
                rows.append(dict(time=float(t), vector=v, value=float(row.iloc[0][v]) + s * rng.standard_normal(), sigma=s))
        return cls(pd.DataFrame(rows))


def misfit(obs: Observations, simulated: pd.DataFrame) -> pd.DataFrame:
    """Simulated against observed: columns time, vector, observed, simulated, residual (simulated - observed),
    normalised (residual / sigma) and weight.  ``simulated`` has the column TIME and the vectors.

    ``rms(...)`` and ``objective(...)`` give a single number; the table shows *which* well, vector and period is off."""
    rows = []
    for r in obs.table.itertuples():
        sim_row = simulated[np.isclose(simulated.TIME, r.time)]
        if sim_row.empty:
            raise ValueError(f"no simulated data at time {r.time}")
        s = float(sim_row.iloc[0][r.vector])
        rows.append(dict(time=r.time, vector=r.vector, observed=r.value, simulated=s,
                         residual=s - r.value, normalised=(s - r.value) / r.sigma, weight=r.weight))
    return pd.DataFrame(rows)


def linear_terms(time, weights: dict) -> pd.DataFrame:
    """Terms of a *linear* objective ``J = sum weight * vector(time)``, for example an economic objective:

        linear_terms(600.0, {"FOPT": -60.0, "FWPT": 5.0, "FWIT": 1.0})     # minus the revenue of the oil, plus the water costs

    Minimising such a J maximises the profit.  Pass the table as ``linear_terms=`` to ``adjoint_gradient`` or
    ``HistoryProblem``; ``evaluate_linear_terms`` gives its value for a table of simulated vectors."""
    return pd.DataFrame([dict(time=float(time), vector=v, weight=float(w)) for v, w in weights.items()])


def evaluate_linear_terms(terms: pd.DataFrame, simulated: pd.DataFrame) -> float:
    """``sum weight * simulated value`` for the terms of ``linear_terms`` (``simulated`` has the column TIME and the vectors)."""
    total = 0.0
    for r in terms.itertuples():
        row = simulated[np.isclose(simulated.TIME, r.time)]
        if row.empty:
            raise ValueError(f"no simulated data at time {r.time}")
        total += r.weight * float(row.iloc[0][r.vector])
    return total


def objective(misfit_table: pd.DataFrame) -> float:
    """The history-matching objective ``J = sum weight * normalised**2`` (Intersect eq. 2.1045: the sum over wells and
    times of the squared error in units of sigma, with the optional weighting factor)."""
    return float(np.sum(misfit_table.weight.to_numpy() * misfit_table.normalised.to_numpy() ** 2))


def rms(misfit_table: pd.DataFrame) -> float:
    """Root mean square of the normalised residuals: about 1 for a model that fits to within the noise."""
    return float(np.sqrt(np.mean(misfit_table.normalised.to_numpy() ** 2)))
