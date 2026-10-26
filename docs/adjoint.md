# Adjoint gradients with kusim

`kusim.adjoint` computes the gradient of an objective (a history-matching misfit, or a profit) with respect to the
permeability, the porosity and the well controls, for the cost of about three simulations, however many variables there
are. `kusim.gradients.HistoryProblem` wraps it with the parameter and control vectors, so that any gradient-based optimiser
can use it. The method follows the adjoint workflow described in `adjoints/ix_adoints.md` (Intersect), adapted to the
IMPES scheme of kusim. Read the section on what the objective looks like before using it for anything important.

```python
problem = HistoryProblem(model, observations, params=ParameterSet([...]),           # ln k, porosity, ...
                         controls=ControlSet([...]), initial_controls={...},         # well targets per control period
                         linear_terms=linear_terms(600.0, {"FOPT": -60.0, ...}),     # an economic term, optional
                         prior=(mean, std))                                          # a Gaussian prior, optional
x0 = problem.vector0()
J, g, result = problem.value_and_gradient(x0)      # by the adjoint
problem.check(x0)                                  # against finite differences through the simulator
scipy.optimize.minimize(lambda x: problem.value_and_gradient(x)[:2], x0, jac=True, method="L-BFGS-B", bounds=...)
```

## From the Intersect equations to this code

| Intersect (`adjoints/ix_adoints.md`) | kusim |
|---|---|
| objective `J = sum (D_sim - D_meas)^2 s / sigma^2` (eq. 2.1045) | `Observations` (with `weight` = `s`), `misfit`, `objective`; `HistoryProblem.value` |
| states `x^n`, equations `g^n(x^{n+1}, x^n, u^n) = 0` | the state is `(p, Sw)`; one step is `A(x^n) p^{n+1} = d(x^n, u^n)` followed by the explicit `S^{n+1} = clip(S^n + F(p^{n+1}, x^n))`. **The coefficients use only the old state**, so the Jacobian with respect to the new state is triangular: there is no Newton iteration |
| adjoint recursion, a backward solve with the transposed Jacobian (eq. 2.1046) | `backward_step`: one **transposed sparse solve** with the pressure matrix per step, then products with the derivatives of the coefficients (the slopes of the property tables). No CPRA preconditioner is needed |
| the Jacobians "reconstructed from saved simulator state information" | the trajectory stores the states and controls only; the sweep recomputes each step's intermediate quantities (`forward_step`) |
| gradient `dJ/du = dL/du + lambda^T dg/du` (eq. 2.1047) | the gradients of the geometry factors, the well indices, the porosity and the targets are accumulated over the steps and turned into `dJ/dkx, dky, dkz, dphi` (`_field_gradients`) and `dJ/d target` |
| control periods, possibly several time steps each | `ControlSet`: one target per control period; the gradients of the steps of a period are summed (`pullback`) |
| linear constraints and bounds handled by the external optimiser | SciPy bounds and equality constraints (`examples/06`); `ParameterSet.bounds`, `ControlSet.bounds` |
| gradients only with respect to cell permeability | here also porosity, well controls and the initial state; not scalar parameters (skin, rock compressibility): use finite differences |
| "the optimiser is not part of Intersect" | the same: SciPy's L-BFGS-B and SLSQP are used in `examples/` |

The code is a **separate, functional copy** of `Simulator.step` (`forward_step`), with its transpose written by hand
(`backward_step`) next to it: the backward pass runs the forward operations in reverse order, each with its local derivative.
The geometry, the well indices and the heads come from a `Simulator` built from the same model; only the arithmetic of the
step is duplicated. The copy reproduces the simulator **exactly** (differences of 0.0 in pressure, saturation and well data),
and the tests keep it so.

## What is differentiated, and what is not

The gradient is that of the algorithm as implemented. Frozen decisions (no derivative through them): the choice of the
upstream block, the saturation clip (a clipped block passes no gradient), the table segment, a producer switching to its
`min_bhp`, a stop at `min_oil_rate`. Outside the PVT table the lookups are flat, and so are their derivatives. The initial
state does not depend on the parameters. Objective vectors: `WBHP`, `WOPR`, `WWPR`, `WWCT`, `WWIR`, `FPRP`, `FOPT`, `FWPT`,
`FWIT`. Both upstream rules (`potential`, `pressure`) and restarts from a `SimState` are supported.

## Evidence

1. **The copy equals the simulator** (bit for bit) for layered, dipping, bottom-hole-pressure, fixed-oil-rate, `min_bhp`
   and `min_oil_rate` cases, under both upstream rules, with schedules (`tests/test_adjoint.py`).
2. **The transpose equals finite differences, step by step.** A dot-product test for every pair of well modes, for the
   state, `kx`, `ky`, `kz`, the porosity and both controls, with Richardson-extrapolated differences: the tests require 1e-5, and the
   agreement is typically 1e-6 or better. (A first check that disagreed at 1e-4 was finite-difference noise: the differences
   converged to the adjoint as the step grew.)
3. **The gradient of a whole run equals finite differences through the simulator**, for the parameters, the controls in every
   mode pair, a schedule, a dipping model, the pressure rule, a restart state, the linear terms, and the prior. At the
   starting point of the twin (72 ln k, 72 logit porosity, 5 controls) most components agree to 1e-6 to 1e-9.
4. **Fifteen deliberate breakages** of the code (a sign, a missing term, the transposition, the clip mask, the cumulative
   volumes, the well pullbacks, the `ky`/`kz` ties, ...) were each caught by a test.

## Cost

On the 72-block twin, one simulation takes 0.046 s and one adjoint value-and-gradient 0.14 s: **3.0 simulations**. A
finite-difference gradient for the 72 log-permeabilities takes **147 simulations** (6.9 s). The adjoint cost does not grow
with the number of variables; the finite-difference cost does.

## The objective is not smooth

This matters more than anything above. The adjoint is the exact derivative of the implemented algorithm *where it is
differentiable*, and the objective of an explicit upwind scheme is not differentiable, or even continuous, in many places.

- **Kinks at tiny scales.** Finite differences of the same gradient component, at steps of 1e-2 to 1e-7 (relative), gave for
  the first injection rate of the twin 3168, 3124, 2985, 1453, 3155.7919 and 3155.7919; the adjoint is 3155.7917. A relative
  step of 1e-4 is under 1 STB/day on a rate of 9000. Finite differences at coarse steps can be off by percent, and at fine
  steps they agree with the adjoint to eight digits. `HistoryProblem.check` therefore tries steps from 1e-4 to 1e-7 and
  accepts a finite-difference estimate only when two neighbouring steps agree; components without one are reported, not hidden.
- **Jumps from the upstream decision.** The upstream block is chosen from the old state while the flux of the saturation update
  uses the new pressures. When the old potential difference passes through zero, the choice flips while the new difference
  is not zero, and the flux jumps. Along the gradient direction in control space the objective jumped by about 377 $ (a relative
  1.4e-6) between points 0.04 STB/day apart, where the slope predicted 254 $.
- **Jumps from the BHP limit.** `min_bhp` is applied after the step as a discrete switch, so the objective is discontinuous in
  the controls. An optimiser of a rate-controlled problem drove the controls onto the edge of the switch: a relative change of
  1e-6 in an injection rate changed the profit from 246.04 to 221.59 M$ (10 %), and the optimiser's own value at its final point was the lower one.
  On the injection-rate problem with the limit active (prices 60 / 20 / 3), the adjoint-based optimiser ended 3 % below a derivative-free
  search at `dt = 10 days` and within 0.04 % at `dt = 5 days` (one run each), which suggests that the size of the jumps depends on the step.
- **Backflow.** A bottom-hole pressure beyond the pressure of the block makes a well flow backwards (a producer that injects).
  The optimiser found that: the first production example shut the producer in by setting its bottom-hole pressure above the
  reservoir pressure, and the oil rate went to -4000 STB/day. kusim does not shut such wells in; `Simulator.diagnostics()` and the
  adjoint diagnostics now count them (`n_backflow_well_steps`, `backflow_well_steps`).
- **One-sided derivatives at table nodes.** A block at connate saturation sits exactly on the first node of the relative-permeability
  and capillary-pressure tables, where the derivative is one-sided (the adjoint uses the right slope: saturation cannot fall below
  connate). Central finite differences of the state there are not comparable.

What follows for a user:

- **Always run the simulator again on a result** (`examples/05` and `06` do), never trust the optimiser's own value.
- **Keep active limits and backflow out of the optimisation**, or make the problem continuous: the production example uses a
  bottom-hole-pressure producer and a rate-controlled injector without a limit, and checks `mode_switches` and the backflow
  counts along the whole path of the optimiser. The proper cure is to apply the limit inside the step (not done).
- **Use `check` with the small steps**, and read `result.diagnostics` (`upstream_flips`, `saturation_clipped`, `mode_switches`,
  `backflow_well_steps`) as a measure of how rough the objective is.
- A smaller `dt` makes the scheme closer to continuous but the run longer.

## The two workflows (`examples/05`, `06`)

**History matching** (`05_adjoint_history_matching.py`), on the twin of the EnKF example. The unknowns are the 72
log-permeabilities, reparameterised as `ln k = mean + L z` with the prior `N(0, I)` on `z`, so that the objective is the data
misfit plus `z.z` and the problem is a maximum a posteriori estimate. L-BFGS-B needed 83 iterations (187 gradient evaluations,
27 s); the data misfit fell from 30.8 sigma to 0.97 sigma (the noise level) and the ln k error from 0.73 to 0.43. With independent
priors on the blocks (no spatial correlation) the estimate is worse (0.63 in the earlier run); the correlation of the prior is what carries the
information from the wells to the rest of the field.

How this compares with the EnKF (`04`) on the same data needs care. The EnKF mean has the same error (0.42) with about 90 simulations
(three runs per member) against about 550 simulation-equivalents here, so the EnKF is cheaper for this small problem. But the adjoint run was given the exact
prior mean and covariance, and the EnKF only a 30-member sample of them; and the adjoint gives one estimate, not a spread. The adjoint
gains on problems with many variables (its cost does not grow) and when the prior is known.

**Production optimisation** (`06_adjoint_production_optimisation.py`). A fixed injection volume (9000 STB/day on average) is
distributed over four periods of 150 days to maximise the profit at 600 days (oil 60, produced water -20, injected water -3 per
STB), with the producer at constant bottom-hole pressure. SLSQP with the adjoint gradient took 14 evaluations and reached
267.93 M$ (uniform: 264.50 M$); Nelder-Mead needed 345 evaluations and reached the same profit, to five digits, with a slightly different
middle allocation: the objective is flat there. The gain is small (1.3 %). The optimised rates inject at the maximum first and
at the minimum last, so that the oil rate collapses at the end: the objective gives no value to what remains in the reservoir
after 600 days (a finite-horizon effect, not a recommendation).

## Not done

- Applying the well limits inside the step, so that the objective does not jump (this changes the simulator).
- Gradients with respect to scalar parameters (skin, rock compressibility, relative-permeability and PVT parameters) and with respect to
  transmissibility multipliers (the model has none yet).
- A Gauss-Newton or other second-order method using the sensitivities of the data rather than only the gradient of the sum of squares.
- Reusing the factorisation of the forward solve in the backward sweep (it factorises the transposed matrix again at every step), and
  large grids: the dense prior of the example does not scale.
- The `linear_solver="gauss"` option of the simulator is not mirrored (the copy uses the sparse solver, which gives the same pressures).
