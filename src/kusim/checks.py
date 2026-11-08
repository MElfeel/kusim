"""Checks before a run: is the model too big, too slow, too large an output, or likely to be unstable?

A person sees a run that does not finish; a program or an agent that launches many runs needs to know **before**.
``check_model`` returns a list of *findings* (``severity`` ``"error"`` means: do not run it as it is, ``"warning"``: run
it, but look), a ``stability`` number and a runtime ``estimate``. ``kusim validate`` shows them, and ``kusim run``,
``misfit``, ``gradient`` and ``ensemble`` refuse to start on an error (``"code": "too_large"``, ``"too_slow"`` or
``"output_too_large"``) and put the warnings in their answer. The limits are arguments and can be raised.

What is estimated, and how far to trust it:

* **Time**: a probe runs the first few steps (default 3) and measures them. The cost of a step grows faster than the
  number of blocks (sparse direct solve): on the machine this was written on, one step took 0.09 s at 10 000 blocks, 0.64 s
  at 32 000, 2.4 s at 72 000 and 12 s at 154 000. The estimate is ``setup + steps * time per step`` from the first steps,
  so it can be off when the machine is busy or the cost per step changes later. The ``"gauss"`` solver,
  a teaching plain-loop elimination, is far slower and is flagged beyond a few hundred blocks.
* **Stability**: IMPES is only conditionally stable, and no single number predicts it. Two signals are used, from the first
  steps of the probe. (1) **The probe itself**: if a saturation had to be moved back into ``[swc, 1 - sor]`` by more than
  ``SATURATION_CLIP_TOLERANCE`` in those steps, the time step is too large. In five families of test cases (the thesis
  input, a layered waterflood with a pressure- or a rate-controlled producer, and the workflows' three-layer model) this
  reproduced the verdict of the whole run whenever the instability starts in the first steps, which is when the injector
  block fills; it fixed the limit at 0.9 to 1.4 for rate-controlled wells. (2) ``Simulator.well_block_throughput``, the
  fraction of the mobile pore volume of a well block that the wells move in one step, above ``COURANT_WARN = 3.3``: this
  catches the thesis case, whose instability starts later (stable up to 2.8, not from 3.5), but not the rate-controlled
  cases, which are unstable from 1.4. The suggested ``dt / k`` is itself probed. **Not covered**: an instability that
  develops later (at water breakthrough, in a high-permeability streak); the ``healthy`` flag of the run is the final word.
* **Output size**: ``results.json`` is estimated from the number of snapshots and well records (about 19 bytes per
  stored number, 350 and 250 bytes per well and perforation record).
* **Memory** (the adjoint stores the whole run) is not estimated.
"""

from __future__ import annotations

import time
from statistics import median

import numpy as np

from .simulator import SATURATION_CLIP_TOLERANCE, Simulator

DEFAULT_MAX_CELLS = 50_000
DEFAULT_MAX_SECONDS = 900.0
DEFAULT_MAX_OUTPUT_MB = 200.0
COURANT_WARN = 3.3
COURANT_TARGET = 2.5                       # the throughput a suggested dt aims for
GAUSS_WARN_CELLS = 400
SUGGEST_FACTORS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32)
_BYTES_PER_NUMBER, _BYTES_PER_WELL_RECORD, _BYTES_PER_PERFORATION_RECORD = 19, 350, 250


def finding(severity, code, message, **details):
    return {"severity": severity, "code": code, "message": message, **details}


def output_megabytes(model, fields="last", save_every=1) -> float:
    """Rough size of ``results.json`` in megabytes (see the module docstring)."""
    n_steps = model.n_steps
    n_saved = {"none": 0, "last": 1, "all": n_steps // max(save_every, 1) + 1}[fields]
    wells = len(model.producers) + len(model.injectors)
    perforations = sum(len(w.blocks) for w in [*model.producers, *model.injectors])
    size = (2 * n_saved * model.grid.n_cells * _BYTES_PER_NUMBER + n_steps * wells * _BYTES_PER_WELL_RECORD
            + n_steps * perforations * _BYTES_PER_PERFORATION_RECORD)
    return size / 1e6


def probe(model, steps=3, upstream="potential", linear_solver="sparse") -> dict:
    """Run the first ``steps`` steps and measure: setup time, time per step, well-block throughput, diagnostics."""
    start = time.perf_counter()
    sim = Simulator(model, linear_solver=linear_solver, save_every=10**9, upstream=upstream)
    setup = time.perf_counter() - start
    times, courant = [], float("nan")
    for _ in range(min(steps, model.n_steps)):
        if sim.finished:
            break
        start = time.perf_counter()
        sim.step()
        times.append(time.perf_counter() - start)
        if len(times) == 1:
            courant = sim.well_block_throughput()
    return dict(setup_seconds=setup, seconds_per_step=median(times) if times else float("nan"),
                well_block_throughput=courant, steps_probed=len(times), diagnostics=sim.diagnostics())


def _unstable(p):
    """The two signals of the module docstring, from a ``probe``."""
    throughput = p["well_block_throughput"]
    return bool(p["diagnostics"]["max_saturation_clip"] > SATURATION_CLIP_TOLERANCE
                or (np.isfinite(throughput) and throughput > COURANT_WARN))


def suggest_dt(model, steps=3, upstream="potential", linear_solver="sparse"):
    """A time step ``dt / k`` (k = 2, 3, ...) at which the first ``steps`` steps show neither signal, by trying them
    (the signals do not scale linearly with dt: the first rates are a transient).
    ``dt / k`` keeps every schedule and observation time of the model on the time grid. Returns ``None`` if no k up to
    ``max(SUGGEST_FACTORS)`` is enough (then the rates are too high for these blocks: lower them or use bigger blocks)."""
    for factor in SUGGEST_FACTORS:
        candidate = model.with_changes(dt=model.dt / factor)
        p = probe(candidate, steps, upstream, linear_solver)
        if not _unstable(p) and not (np.isfinite(p["well_block_throughput"]) and p["well_block_throughput"] > COURANT_TARGET):
            return model.dt / factor
    return None


def _pressure_findings(model):
    lo, hi = float(model.pvt.pressure[0]), float(model.pvt.pressure[-1])
    out = []
    for name, wells in (("P", model.producers), ("I", model.injectors)):
        for n, w in enumerate(wells, 1):
            for what, value in (("bottom-hole pressure target", w.target if w.mode == "bhp" else None),
                                ("min_bhp", getattr(w, "min_bhp", None))):
                if value is not None and not lo <= value <= hi:
                    out.append(finding("warning", "pressure_outside_pvt_range",
                                       f"{name}{n}: {what} {value:g} psi is outside the PVT table ({lo:g} to {hi:g} psi); "
                                       "the properties are extrapolated flat and the result is not reliable",
                                       well=f"{name}{n}", value=value, pvt_range=[lo, hi]))
    return out


def inert_settings(model):
    """Findings for settings that are valid but change nothing, so that nobody believes they acted.

    Today: ``min_bhp`` on a producer in mode ``bhp`` (its pressure is ``target``; ``min_bhp`` only limits a producer on a rate mode)."""
    out = []
    for n, w in enumerate(model.producers):
        if w.mode == "bhp" and w.min_bhp is not None:
            out.append(finding("warning", "inert_setting",
                               f"P{n + 1}: this well is in mode 'bhp', so its bottom-hole pressure is its 'target' ({w.target:g} psi) and "
                               f"min_bhp ({w.min_bhp:g} psi) has no effect; to change the pressure set producers.{n}.target (or P{n + 1}.target)",
                               well=f"P{n + 1}", path=f"producers.{n}.min_bhp"))
    return out


def check_model(model, probe_steps=3, max_cells=DEFAULT_MAX_CELLS, max_seconds=DEFAULT_MAX_SECONDS, fields="last",
                save_every=1, max_output_mb=DEFAULT_MAX_OUTPUT_MB, upstream="potential", linear_solver="sparse",
                n_steps=None, repeats=1) -> dict:
    """Findings, stability and estimate for running ``model`` (see the module docstring).

    ``n_steps`` is the number of steps that will be taken (default: to ``t_end``); ``repeats`` how many runs of that
    length are planned (an ensemble: members times passes; the adjoint: about 3). A limit of 0 (or ``None``) switches it
    off. The probe is skipped (``probe_steps=0``) or not run when the model is over ``max_cells``.
    Returns ``{"runnable", "findings", "stability", "estimate", "output_mb"}``."""
    n_steps = model.n_steps if n_steps is None else n_steps
    findings = _pressure_findings(model) + inert_settings(model)
    n_cells = model.grid.n_cells
    if max_cells and n_cells > max_cells:
        findings.append(finding("error", "too_large", f"{n_cells} blocks is over the limit of {max_cells} "
                                "(a step takes seconds beyond about 50 000 blocks); raise it with --max-cells",
                                n_cells=n_cells, limit=max_cells, flag="--max-cells"))
    if linear_solver == "gauss" and n_cells > GAUSS_WARN_CELLS:
        findings.append(finding("warning", "slow_solver", f"the 'gauss' solver is a teaching plain-loop elimination "
                                f"(O(n^3)); with {n_cells} blocks use the default 'sparse'", n_cells=n_cells))
    output_mb = output_megabytes(model, fields, save_every)
    if max_output_mb and output_mb > max_output_mb:
        findings.append(finding("error", "output_too_large", f"results.json would be about {output_mb:.0f} MB (limit "
                                f"{max_output_mb:g}); use --fields last or none, --save-every N, or raise --max-output-mb",
                                output_mb=output_mb, limit=max_output_mb, flag="--max-output-mb"))
    stability = estimate = None
    if probe_steps and not any(f["code"] == "too_large" for f in findings):
        try:
            p = probe(model, probe_steps, upstream, linear_solver)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            findings.append(finding("error", "probe_failed", f"the first steps could not be run: {error}"))
        else:
            throughput, clip = p["well_block_throughput"], p["diagnostics"]["max_saturation_clip"]
            unstable = _unstable(p)
            stability = {"well_block_throughput": throughput, "warn_above": COURANT_WARN,
                         "max_saturation_clip_first_steps": clip, "steps_probed": p["steps_probed"],
                         "likely_stable": not unstable}
            if unstable:
                suggested = suggest_dt(model, max(probe_steps, 3), upstream, linear_solver)
                stability["suggested_dt"] = suggested
                advice = (f"try dt = {suggested:.3g} days (a fraction of the present dt, so schedule and observation "
                          "times stay valid)" if suggested else "no smaller dt of the form dt/k up to k = 32 is enough: "
                          "lower the rates or use bigger blocks")
                reason = (f"a saturation had to be moved back by {clip:.2g} in the first {p['steps_probed']} steps"
                          if clip > SATURATION_CLIP_TOLERANCE else
                          f"the wells move {throughput:.1f} times the mobile pore volume of a well block per step")
                findings.append(finding(
                    "warning", "dt_too_large", f"{reason}: the explicit saturation update is likely to oscillate; {advice}",
                    well_block_throughput=throughput, max_saturation_clip=clip, dt=model.dt, suggested_dt=suggested))
            if p["diagnostics"]["outside_pvt_range"] and not any(f["code"] == "pressure_outside_pvt_range" for f in findings):
                findings.append(finding("warning", "pressure_outside_pvt_range",
                                        "the pressure left the PVT table in the first steps; the properties are "
                                        "extrapolated flat and the result is not reliable",
                                        p_min=p["diagnostics"]["p_min"], p_max=p["diagnostics"]["p_max"]))
            one_run = p["setup_seconds"] + n_steps * p["seconds_per_step"]
            estimate = {"setup_seconds": p["setup_seconds"], "seconds_per_step": p["seconds_per_step"],
                        "n_steps": n_steps, "repeats": repeats, "seconds": one_run * repeats,
                        "seconds_one_run": one_run, "basis": f"{p['steps_probed']} steps timed"}
            if max_seconds and estimate["seconds"] > max_seconds:
                findings.append(finding(
                    "error", "too_slow", f"estimated {estimate['seconds']:.0f} s ({repeats} run(s) of {n_steps} steps at "
                    f"{p['seconds_per_step']:.3g} s per step) is over the limit of {max_seconds:g} s; raise it with "
                    "--max-seconds, or use a shorter time, a larger dt or fewer blocks", seconds=estimate["seconds"],
                    limit=max_seconds, flag="--max-seconds"))
    return {"runnable": not any(f["severity"] == "error" for f in findings), "findings": findings,
            "stability": stability, "estimate": estimate, "output_mb": output_mb}
