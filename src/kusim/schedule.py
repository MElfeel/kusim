"""Well controls that change with time (a schedule), and a way to run a simulator through it.

A history is a list of control changes: "from day 200 well P1 produces 6000 STB/day", "from day 400 well I1 is
controlled at 4800 psi".  A ``Schedule`` holds such events and ``advance`` applies them at the right steps:

    schedule = Schedule([ControlEvent(0, "P1", "liquid_rate", 8000.0),
                         ControlEvent(200, "P1", target=6000.0),             # same mode, new target
                         ControlEvent(400, "I1", "bhp", 4800.0)])
    sim = Simulator(model)
    advance(sim, 600, schedule)        # steps to day 600, changing the controls on day 200 and 400

The wells start with the controls of the model; an event changes the mode, the target or both.  To close a well
for a while, give it a rate of 0.  Event times (and the end time of ``advance``) must be multiples of the time step,
because the simulator takes fixed steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlEvent:
    """From ``time`` (days) well ``well`` gets ``mode`` and/or ``target`` (``None`` keeps the present value)."""

    time: float
    well: str
    mode: Optional[str] = None
    target: Optional[float] = None

    def __post_init__(self):
        if self.time < 0:
            raise ValueError("an event time must not be negative")
        if self.mode is None and self.target is None:
            raise ValueError(f"the event of well {self.well} at day {self.time} changes nothing")


class Schedule:
    """A list of control events, kept in time order."""

    def __init__(self, events: Iterable[ControlEvent] = ()):
        self.events = sorted(events, key=lambda e: e.time)      # stable: events at the same time keep their order

    @classmethod
    def from_table(cls, table):
        """From a table (pandas DataFrame or list of dicts) with the columns time, well and mode and/or target."""
        rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
        events = []
        for r in rows:
            mode, target = r.get("mode"), r.get("target")
            events.append(ControlEvent(float(r["time"]), r["well"],
                                       None if mode is None or mode != mode else mode,        # mode != mode: NaN
                                       None if target is None or target != target else float(target)))
        return cls(events)

    def with_event(self, event: ControlEvent) -> "Schedule":
        """A new schedule with one more event."""
        return Schedule([*self.events, event])

    def times(self):
        """The distinct event times."""
        return sorted({e.time for e in self.events})

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        return f"Schedule({len(self.events)} events, wells {sorted({e.well for e in self.events})})"


def plan_advance(t0, until, dt, t_end, schedule: Optional[Schedule] = None, tol=1e-6):
    """Check a run from ``t0`` to ``until`` and list what happens: ``(steps, pending)`` where ``pending`` are the
    events (in time order) to apply on the way.  Raises ValueError for times that are not multiples of ``dt``."""
    if until > t_end + tol:
        raise ValueError(f"until = {until} is beyond the end of the model, t_end = {t_end}")
    steps = (until - t0) / dt
    if steps < -tol or abs(steps - round(steps)) > tol:
        raise ValueError(f"from day {t0} to day {until} is not a whole number of steps of {dt} days")
    pending = [e for e in (schedule.events if schedule else []) if e.time >= t0 - tol and e.time < until - tol]
    for e in pending:
        k = e.time / dt
        if abs(k - round(k)) > tol:
            raise ValueError(f"the event of well {e.well} at day {e.time} is not at a multiple of dt = {dt}")
    return int(round(steps)), pending


def advance(sim, until, schedule: Optional[Schedule] = None, tol=1e-6):
    """Run ``sim`` from its present time to day ``until``, applying the events of ``schedule`` on the way.

    An event is applied before the first step that starts at or after its time, so an event at day 200 governs
    the step from day 200 to 200 + dt.  Events before the present time are not applied again (the state of a
    restarted simulator already holds the controls then in force); events at the present time are.
    Returns the number of steps taken (fewer than asked if a stop condition of a well ended the run).
    """
    steps, pending = plan_advance(sim.time, until, sim.model.dt, sim.model.t_end, schedule, tol)
    done = used = 0
    for _ in range(steps):
        while used < len(pending) and pending[used].time <= sim.time + tol:
            e = pending[used]
            sim.set_well_control(e.well, e.mode, e.target)
            used += 1
        sim.step()
        done += 1
        if sim.stop_reason:
            break
    return done


# ---------------------------------------------------------------------- controls as optimisation variables
@dataclass
class ControlParameter:
    """The targets of one well as optimisation variables: one number per control period.

    ``times`` are the start times (days) of the periods; the target of a period holds until the next one starts.
    ``mode`` (optional) is set with every target.  ``lower`` and ``upper`` bound the target (rates are not negative).
    """

    well: str
    times: Sequence[float]
    mode: Optional[str] = None
    lower: Optional[float] = 0.0
    upper: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.times = [float(t) for t in self.times]
        if not self.times or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError(f"{self.well}: the period start times must increase")
        self.name = self.name or f"{self.well}_targets"


class ControlSet:
    """A vector of control targets and the schedule it stands for.

        controls = ControlSet([ControlParameter("I1", times=[0, 200, 400], mode="water_rate", upper=12000.0)])
        schedule = controls.schedule(np.array([9000.0, 8000.0, 6000.0]), base=history)

    ``pullback`` turns the gradient with respect to the target of every *step* (what the adjoint gives) into the
    gradient with respect to the vector (one number per period).
    """

    def __init__(self, parameters: Sequence[ControlParameter]):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("a ControlSet needs at least one parameter")
        wells = [p.well for p in self.parameters]
        if len(set(wells)) != len(wells):
            raise ValueError(f"one ControlParameter per well, got {wells}")

    @property
    def size(self):
        return sum(len(p.times) for p in self.parameters)

    def _slices(self):
        out, start = {}, 0
        for p in self.parameters:
            out[p.well] = slice(start, start + len(p.times))
            start += len(p.times)
        return out

    def bounds(self):
        """``(lower, upper)`` vectors (infinite where a parameter has no bound)."""
        lower = np.concatenate([np.full(len(p.times), -np.inf if p.lower is None else p.lower) for p in self.parameters])
        upper = np.concatenate([np.full(len(p.times), np.inf if p.upper is None else p.upper) for p in self.parameters])
        return lower, upper

    def vector(self, targets: dict):
        """The vector for ``targets = {well: [target of period 0, 1, ...]}``."""
        parts = []
        for p in self.parameters:
            values = np.asarray(targets[p.well], dtype=float)
            if values.shape != (len(p.times),):
                raise ValueError(f"{p.well}: {len(p.times)} targets are needed, got {values.size}")
            parts.append(values)
        return np.concatenate(parts)

    def schedule(self, vector, base: Optional[Schedule] = None) -> Schedule:
        """The schedule of the targets in ``vector`` (clipped to the bounds).  Events of ``base`` for other wells are kept;
        those for the controlled wells are replaced."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(f"the vector has {vector.size} values, the controls need {self.size}")
        slices = self._slices()
        controlled = set(slices)
        events = [e for e in (base.events if base else []) if e.well not in controlled]
        lower, upper = self.bounds()
        clipped = np.clip(vector, lower, upper)
        for p in self.parameters:
            for t, target in zip(p.times, clipped[slices[p.well]]):
                events.append(ControlEvent(t, p.well, p.mode, float(target)))
        return Schedule(events)

    def pullback(self, vector, step_gradients: dict, step_times, tol=1e-6):
        """Gradient with respect to ``vector`` from ``step_gradients = {well: dJ/d(target) of every step}``.

        ``step_times`` are the start times of the steps.  A step belongs to the period that is in force at its start; a
        period whose target is clipped by a bound has zero gradient."""
        vector = np.asarray(vector, dtype=float)
        lower, upper = self.bounds()
        free = ((vector >= lower) & (vector <= upper)).astype(float)
        step_times = np.asarray(step_times, dtype=float)
        out = np.zeros(self.size)
        for p in self.parameters:
            sl = self._slices()[p.well]
            period = np.searchsorted(np.asarray(p.times), step_times + tol, side="right") - 1
            g = np.asarray(step_gradients.get(p.well, np.zeros(len(step_times))), dtype=float)
            for j in range(len(p.times)):
                out[sl.start + j] = g[period == j].sum()
        return out * free
