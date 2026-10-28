# History matching and ensemble methods with kusim

kusim has a small, public interface for programs that **edit models, run many of them, restart them and compare
them with data**: ensemble methods (the Ensemble Kalman Filter, ES-MDA, ...), optimisation loops, and agents that
build and change models. Two ready-made methods, **ES-MDA** and the **EnKF**, are in `kusim.ensemble` (the analysis
step, localisation, a Gaussian prior sampler) and run from the command line with `kusim ensemble` (see
`docs/json-format.md`); the modules of `examples/` (`enkf_reference.py`, `esmda.py`) re-export them for the course. Other
methods (EnKS, iterative smoothers, a different localisation) are meant to be written on the interface below, as the
examples show.

| Module | What it gives |
|---|---|
| `ReservoirModel.with_changes` | an edited **copy** of a model (arrays, nested values, wells), with a log of the edits |
| `kusim.parameters` | `ParameterSet`: the model as a **vector** (log-permeability, ...) and back |
| `Simulator.get_state / set_state` | save and restore the dynamic state: **restart**, also in an edited model |
| `Simulator.set_well_control`, `kusim.schedule` | well controls that change with time |
| `kusim.history` | summary vectors, observations, running members (`run_to`, `run_ensemble`), misfit, linear (economic) terms |
| `kusim.adjoint`, `kusim.gradients`, `ControlSet` | **gradient-based** matching and optimisation: the adjoint gradient of an objective with respect to permeability, porosity and well controls, an alternative to ensemble methods: see `docs/adjoint.md` |
| `Simulator.diagnostics`, `RunRecord.ok` | signs that a run is not to be trusted |

## The Ensemble Kalman Filter step by step (`enkf/HA_localised_enkf.docx`)

| In the filter | In kusim |
|---|---|
| prior ensemble of N models | a list of `ReservoirModel`; `ParameterSet.apply_all(base, matrix)` builds them from vectors |
| augmented state `y = [static m, dynamic p, S, simulated data d]` | `params.matrix(models)` (static, in log space), `record.state.p_o`, `record.state.s_w`, `data_matrix(records, t, vectors)` |
| forecast: run each member to the next measurement date from its own state | `run_ensemble(models, [t], states=states)` |
| observed data with error | `Observations` (`time, vector, value, sigma`); `obs.at(t)` |
| analysis (Kalman gain, localisation by a Schur product) | `kusim.ensemble.enkf_analysis`; `localisation_yd` builds the distance matrix from `grid.cell_centres()` and the well blocks |
| write the updated parameters back | `params.apply(model, y_updated)`: transforms back, clips to bounds, keeps `ky`, `kz` tied to `kx` |
| restart from the updated pressure and saturation | `dataclasses.replace(state, p_o=..., s_w=...)`, then `run_to(..., state=state)`; saturations are clipped to `[swc, 1 - sor]` and the count is in `record.adjusted` |
| consistency: rerun from time zero (HIEnKF, EnKS) | `run_ensemble(models, all_dates)` without states; compare with the data using `misfit` |
| non-3D uncertain parameters (WOC, skin, relperm, ...) | `ScalarParameter("initial.woc_depth")`, `ScalarParameter("producers[0].skin")`; a changed WOC needs the rerun from time zero |

## Using the pieces

```python
from kusim.parameters import FieldParameter, ScalarParameter, ParameterSet
from kusim.history import run_ensemble, data_matrix, Observations, misfit, rms
from kusim.schedule import Schedule, ControlEvent

params = ParameterSet([
    FieldParameter("kx", "log", lower=1.0, upper=1e4, ties=("ky", "kz")),     # one number per block
    FieldParameter("porosity", "logit", lower=0.02, upper=0.40),
    ScalarParameter("initial.woc_depth"),                                      # one number
])
y = params.vector(model)                       # transformed space: ln k, logit(phi), woc
better = params.apply(model, y + dy)           # a new model; `model` is unchanged; better.provenance logs the edit

history = Schedule([ControlEvent(0, "P1", "liquid_rate", 8000), ControlEvent(200, "P1", target=6000),
                    ControlEvent(400, "I1", "bhp", 4800)])
records = run_ensemble(members, [50, 100], history, vectors=["WBHP:P1", "WWCT:P1"], workers=4)
D = data_matrix(records, 100, ["WBHP:P1", "WWCT:P1"])          # (data, members)
next_records = run_ensemble(members, [150], history, states=[r.state for r in records])
```

Vector names are the usual summary vectors: `WBHP:P1`, `WOPR:P1`, `WWPR:P1`, `WWCT:P1`, `WWIR:I1`, `FPRP`, `FOPT`,
`FWPT`, `FWIT`. `kusim.history.misfit(obs, table)` gives the residual of every datum in units of its sigma, so that
you can see *which* well, vector and period is off.

## Rules that are easy to get wrong

- **Models are copied, never edited in place.** `with_changes` deep-copies; nothing is shared with the old model.
  Changing the object that you passed in afterwards does not change the copy.
- **Times are multiples of `dt`.** The simulator takes fixed steps: schedule events, report times and restart states
  must be at multiples of `dt` (a clear error is raised otherwise). Rates reported at a date are those of the step that
  ends there.
- **A state holds more than p and S**: the controls in force (a producer that reached its `min_bhp` has switched to
  "bhp") and the cumulative volumes. A restart in another model continues with these.
- **An update can leave the physical range.** `set_state` puts saturations back into `[swc, 1 - sor]` (and says
  how many, `record.adjusted`); pressures are not changed, but `diagnostics()` flags a run that leaves the PVT table.
  `ParameterSet.apply` keeps parameters inside their bounds (log and logit transforms keep them positive or inside
  their limits for any additive update).
- **Look at `RunRecord.ok` and the diagnostics.** A member whose pressure left the PVT table, or whose saturation had
  to be clipped often, gives a misfit that says nothing about its parameters. In the twin experiment a very heterogeneous
  truth (ln k standard deviation 2) reached 10 000 psia, beyond the table at 5 500, and was rejected.
- **The static and dynamic variables are updated separately by the filter**, so the updated state is not consistent
  with the updated model. Run the updated members from time zero to see how well the *models* reproduce the history.
- **Several processes (`workers > 1`, or an `executor`)** copy the models to the workers. Starting processes takes
  seconds on Windows, so make one `ProcessPoolExecutor` for a whole loop; scripts need `if __name__ == "__main__":`.

## What the twin experiment shows (`examples/04_enkf_twin_experiment.py`)

A 6 x 4 x 3 layered model (the bonus project), a truth with a correlated log-permeability field, noisy producer and
injector pressure (sigma 10 psi) and producer water cut (sigma 0.02) every 50 days for 600 days, 40 prior members, no
localisation. The whole loop takes a few seconds.

- **The data are matched.** The residual in units of sigma (mean over members) falls from about 50 to about 1.5 after
  the updated members are re-run from time zero, on every one of 16 seeds (26 to 69 before, 1.4 to 15 after).
- **The permeability field is recovered only partly.** The error of the ensemble mean against the truth fell on 10 of
  16 seeds at the cell level and on 12 of 16 for the three layer averages; the mean error of the middle (flow-controlling)
  layer fell from 0.42 to 0.18 in ln k. Two wells and three data types say little about 72 blocks; the ensemble spread
  halves (0.7 to 0.4), which is more confident than the error justifies. This is the small-ensemble behaviour that the
  localisation of the document addresses, and it makes the twin a good test bed for it.
- **A short localisation length restricts the update**, as it should (`localise_length=500` ft leaves every block
  beyond 1000 ft of both wells exactly at its prior value), and then matches the data less well. Choosing the
  length is the research question; the hook is in place.

## ES-MDA: the method of the course (`examples/07_esmda_twin_experiment.py`)

The permeability does not change with time, only what we know about it does, so for history matching of static parameters a
**smoother** is the natural choice: run every member over the whole history, compare with all the data at once, update the
parameters, and run again from day 0. A single update on a nonlinear problem is too large a step, so **ES-MDA** (ensemble
smoother with multiple data assimilation) makes several, each pretending that the measurement error is larger by a factor
`sqrt(alpha)`, with `sum(1/alpha) = 1` so that the data are used exactly once overall. The update is the EnKF analysis of
`kusim.ensemble.enkf_analysis` with `sigma * sqrt(alpha)`, applied to the parameters only (`esmda_update`; `run_esmda` is
the whole loop, `kusim ensemble --method esmda` its command); the twin workflow is `esmda_twin.py` (same truth, data and prior as the EnKF twin). There are no restarts and no update of pressure or saturation,
so the models are consistent with their parameters by construction.

Measured on the same twin (40 members, seeds 0 to 15, 4 passes of `alpha = 4`; the EnKF on the same seeds for comparison):

| | ES-MDA | EnKF (`04`) |
|---|---|---|
| data misfit after the update, in sigma (mean over members), mean and range over seeds | 1.5 (1.1 to 2.6) | 3.6 (1.3 to 11.7) |
| ln k error of the ensemble mean, mean over seeds, prior to updated | 0.66 to 0.59 | 0.66 to 0.58 |
| seeds on which the ln k error fell | 11 of 16 | 10 of 16 |
| ensemble spread of ln k, mean over seeds, prior to updated | 0.71 to 0.34 | 0.71 to 0.39 |
| simulations per member | 5 (the prior and one per pass) | about 3 |
| time, 40 members, one core | about 10 s | about 6 s |

- **The data are matched slightly better and more evenly, the field about as well.** The permeability field is recovered only
  partly by both; on seed 12 the ES-MDA field error got worse (0.55 to 1.10) while the data were matched (1.5 sigma). A good data
  match is not a good field.
- **The spread collapses a little more than with the EnKF** (0.34 against 0.39) and is again more confident than the error justifies.
- **The first pass can push members out of the valid range.** On seed 1, 30 of 40 members left the PVT table after the first pass
  (`RunRecord.ok` false) and the next passes brought them back. Read the diagnostics after every pass (`metrics["not_ok_by_pass"]`);
  a member that is not ok has a misfit that says nothing about its parameters.
- **No localisation.** Distance localisation of the state-data covariance alone (the `enkf_twin.localisation_yd` weights, which work
  in the EnKF) made the ES-MDA update overshoot: at lengths of 500, 800, 1200 and 2000 ft (seed 0, 30 members) the change of ln k
  reached 29 in one block in the second pass and the simulator returned NaN in the third. With the data of all dates stacked, weighting each
  datum differently per block breaks the cancellation between the injector and producer data. A consistent localisation is
  future work; the ES-MDA code has none.
- The number of passes and the factors are choices (equal factors here, found to work on this twin; not tuned).

## Not done (next steps)

- Only ES-MDA and the EnKF are in the package (`kusim.ensemble`); EnKS, iterative smoothers and other localisations are not. The prior is Gaussian in the transformed space with an exponential correlation; correlated fields use a dense covariance and are limited to 3000 blocks.
- **Physics the model cannot represent**, so a match will absorb its absence in the permeability: no transmissibility
  multipliers (faults), no regions or per-region relative permeability, no aquifer, no gas. See `docs/porting-notes.md`.
- **Fixed time step** (IMPES): there is no adaptive step and no warning when `dt` is too large for stability; the only
  signs are the saturation-clip count and oscillating well data.
- **Observation times that are not multiples of `dt`** need a variable last step; not supported.
- No batch tool for sensitivity or priors beyond what the example does, and no checkpoint-restart cache by model hash.
- Parameter fields are one value per block; a smaller parameterisation (zones, pilot points, KL expansion) is up to the caller.
