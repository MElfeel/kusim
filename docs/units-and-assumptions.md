# Units and assumptions

One page for what a model file means. For the file formats see `docs/json-format.md`; for where the code differs from
the thesis and the FORTRAN see `docs/porting-notes.md`; for the equations `docs/05-equations.md`.

## Units

kusim works in **oilfield units only**. There is no unit conversion anywhere: convert before you write a file.

| Quantity | Unit | Notes |
|---|---|---|
| lengths (`dx`, `dy`, `dz`, depth, well radius) | ft | depth is positive downward; `top_depth` is the depth of the top of layer 1 |
| permeability (`kx`, `ky`, `kz`) | md | |
| porosity, saturations, relative permeabilities | fraction (0 to 1) | water cut too (the injector's is `null`) |
| pressure (initial, tables, targets, results) | psi | the code does not tell absolute from gauge: use the same basis as the PVT table |
| viscosity (`mu_o`, `mu_w`) | cp | |
| formation volume factors (`bo`, `bw`) | rb/STB | the code stores and interpolates **1/B** internally |
| compressibility (rock) | 1/psi | |
| capillary pressure (`pcow`) | psi | `Pcow = Po - Pw` |
| specific gravity (`gamma_o`, `gamma_w`) | relative to water = 1 | the gradient is 0.433 psi/ft per unit |
| rates (`target` of a rate-controlled well; results) | STB/day (surface) | **positive for producers and injectors** |
| cumulative volumes | STB | |
| time (`dt`, `t_end`, schedule and observation times) | days | all must be multiples of `dt` |
| volumes inside the simulator | ft³ | well terms carry a 5.615 ft³/bbl factor (porting-notes decision 1) |

Block numbers in files count **from 1**: x fastest, then y, then z, top layer first
(`block = (k-1)*nx*ny + (j-1)*nx + i`). Wells are named `P1, P2, ...` and `I1, I2, ...` in the order of the file.
Positions in lists inside `kusim edit` paths count from 0 (`producers.0` is `P1`).

## What the model represents

- **Two phases**, oil and water. The compressibility of the fluids comes from the slope of `1/B` in the PVT table;
  the rock has one constant compressibility. No gas, no dissolved gas.
- **Single PVT region**, one rock-fluid table (relative permeability and capillary pressure against Sw) for the whole
  grid. No hysteresis.
- **Rectangular block grid**: `dx` per column, `dy` per row, `dz` per layer; blocks are boxes, optionally on a dipping
  top (`top_depth` per column). Transmissibility uses the harmonic average of the permeabilities of two blocks.
- **Closed outer boundary**: no flow across the faces of the grid, no aquifer. Fluid enters and leaves only through
  wells (and by expansion).
- **Porosity** is constant in the pore volume; rock compressibility enters only the storage term.
- **Gravity and capillary pressure** are included. The initial state is either **uniform** (one pressure and one Sw)
  or **equilibrium** (capillary-gravity equilibrium from a pressure at a datum depth and a water-oil contact, all depths in
  the frame of the grid; needs a capillary pressure table).
- **Saturations stay in `[swc, 1 - sor]`**: a value pushed outside by the explicit update is put back and counted
  (`n_saturation_clipped`, and `max_saturation_clip`, the largest move). Tiny moves (about 1e-4) occur at connate saturation where
  the oil expands near a producer and are harmless; moves of tenths mean an unstable time step. The run is `healthy` only if
  no saturation was moved by more than 0.01.

## Wells

- **Control modes.** Producer: `oil_rate`, `liquid_rate` (oil plus water), `bhp`. Injector: `water_rate`, `bhp`.
  `target` is STB/day for rate modes and psi for `bhp`.
- **Peaceman** well index per perforation, from the block's `kx` and `ky`, the well `radius` (ft) and `skin`.
- **Several layers.** A well can be completed in a list of blocks. A `bhp` well flows per perforation against its own
  pressure; a rate-controlled well shares its rate by well index times mobility (explicit, so it cannot cross-flow). The
  bottom-hole pressure is at the shallowest perforation; the wellbore fluid weight (`wellbore_gravity`, default oil for a
  producer and water for an injector) is included.
- **Limits.** `min_bhp`: a rate-controlled producer that reaches it switches to `bhp` at that pressure. `min_oil_rate`:
  the run **stops** when the oil rate falls to it (`stop_reason`).
- **No shut-in logic.** A well whose bottom-hole pressure passes the block pressure flows backwards (a producer injects).
  The run counts it (`n_backflow_well_steps`); an optimiser can exploit it.

## Numerical scheme

- **IMPES**: implicit pressure, explicit saturation, one pressure solve per step, first-order **upstream** mobility.
  The upstream block of a phase is chosen by its **potential** (pressure minus gravity head), not by pressure alone
  (porting-notes decision 18); `--upstream pressure` gives the thesis rule, which traps sinking water in vertical flow.
- **Conditionally stable.** Large rates, small blocks or a large `dt` make the saturation update oscillate. Before a
  run, `kusim validate` runs the first 3 steps and warns (`dt_too_large`, with a suggested `dt`) if a saturation had to be moved
  back by more than 0.01 in them, or if the wells move more than 3.3 times the mobile pore volume of a well block per step
  (`well_block_throughput`). The first signal caught every case whose instability starts at the beginning (the injector block
  filling), including rate-controlled wells where the throughput alone would miss it (unstable from about 1.4); the second
  catches the thesis input, whose instability starts later. Both are heuristics checked on a handful of models: an
  instability that develops later (water breakthrough, a high-permeability streak) is not covered, so the `healthy` flag of the
  run is the final word.
- **Direct sparse solver** by default; the plain-loop `gauss` solver is for teaching and is far slower.
- **Time step is fixed** (`dt`), no adaptive stepping. Events, observations and report times are on the step grid.

## Known limits of the answers

- **Outside the PVT table** the property lookups are flat but the compressibility term keeps the end slope, so mass is
  not conserved there (about 1.5 % of the oil in place after 600 days in one test). Keep pressures inside the table;
  `outside_pvt_range` and a warning from `validate` say when they are not.
- **Not a validated commercial simulator.** The code was checked against Buckley-Leverett, material balance, a
  comparison with ECLIPSE on one case (`eclipse_comparison/`) and the first-step well rates of the legacy input.
  The original FORTRAN executable and output files could not be used as a reference.
- **History matching** recovers the permeability field only partly: the fit to the data improves, the field less
  reliably (`docs/history-matching.md`).
- **The adjoint gradient** is of a scheme with kinks and jumps: check it against finite differences
  and run the simulator on any optimised result.

## Guardrails before a run

`kusim validate`, and the start of `run`, `misfit`, `gradient` and `ensemble`, check the model (`kusim.checks`):

| Finding | Severity | Meaning | Flag |
|---|---|---|---|
| `too_large` | error | more blocks than `--max-cells` (default 50 000; a step takes seconds beyond that) | `--max-cells` |
| `too_slow` | error | estimated time above `--max-seconds` (default 900); the estimate is from timing the first 3 steps, times the steps and the runs planned (members times passes for an ensemble, about 3 for the adjoint) | `--max-seconds` |
| `output_too_large` | error | estimated `results.json` above `--max-output-mb` (default 200) | `--max-output-mb`, `--fields`, `--save-every` |
| `dt_too_large` | warning | one of the two stability signals above; carries `suggested_dt` (a fraction `dt/k`, so existing times stay valid) | |
| `pressure_outside_pvt_range` | warning | a bottom-hole target or `min_bhp` is outside the table, or the pressure left it in the first steps | |
| `slow_solver` | warning | the `gauss` solver on more than 400 blocks | |
| `probe_failed` | error | the first steps could not be run | |

A refused run is exit code 2 with `code` set to the finding and `findings` and `estimate` in the answer. `0` switches a
limit off. Not covered: memory (the adjoint stores the whole run), and a wall-clock timeout (use the timeout of whatever
starts the process).
