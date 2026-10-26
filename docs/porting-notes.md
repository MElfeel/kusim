# Porting notes: where KUSIM-Python follows the FORTRAN and where it does not

The Python package `kusim` implements the IMPES method of Chapter 5, using the first version of the
FORTRAN program (`legacy/fortran-3d3/3DTP(11).FOR`) as the reference for details. The rule for conflicts
between the thesis chapters and the code was: **keep the physics of the thesis, keep the numerical
scheme of the code, and fix anything that is clearly a mistake**, with the aim of a course that is
easy to follow. This page records each decision.

## Mapping of the code to the thesis

| Thesis (Chapter 5) | FORTRAN (`3DTP(11).FOR`) | Python |
|---|---|---|
| 5.2 B, eq. 5-45 to 5-53, storage coefficients | `DO 28` (line 464) | `Simulator.step`, "storage coefficients" |
| 5-16 to 5-27, transmissibilities | `DO 27` (line 445) | `Simulator._build_geometric_factors` and `step` |
| 5-28 to 5-33, upstream mobility | `DO 31` (line 313) | `step`, the `np.where(self._neighbour_is_upstream(...), ...)` lines, and `Simulator._neighbour_is_upstream` (**by potential**, decision 18) |
| gravity terms | `DO 32` (line 476) | `step`, `gra_o`, `gra_w` |
| 5-58 to 5-66, pressure equation and wells | `DO 29`, `DO 30` (lines 569-654) | `step`, `_add_well_terms` |
| 5.5, Gaussian elimination | lines 661-718 | `solver.gauss_elimination` (and sparse solver) |
| 5-68, saturation update | `DO 503` (line 791) | `step`, last block |
| 5-94 to 5-97, initialization | lines 223-243 | `Simulator.reset` |
| `INPUT.DAT` reading | lines 54-121 | `kusim.io.read_legacy_input` |
| `SUBROUTINE INTERP` | line 811 | `kusim.tables.Table`, `invert_decreasing` |

## Decisions where the thesis and the code disagree, or the code has a mistake

1. **Well terms were 5.615 times too small in the FORTRAN (changed).** The flow terms use
   `0.006328 = 0.001127 * 5.615` (ft³ basis) and the storage terms are ft³/ft³, but the well terms are
   `Q / cell volume` with Q in STB/day, with no ft³/bbl factor. Python multiplies the well terms by 5.615.
   *Consequence:* wells are physically stronger than in the old program. The thesis test case (Application 1:
   5000 STB/day from a 50 ft block, rock compressibility 4.5e-4) then needs `dt` of about 0.1 day or less;
   at the file's `dt = 1` day the explicit saturation update oscillates. See `examples/02_legacy_case.py`.
2. **Injector well equation (changed).** The FORTRAN uses `BO/BW` after both had been inverted, which is
   the ratio upside down. Python uses `(Bo/Bw) * lam_o + lam_w` as in eq. 5-74.
3. **Producer water rate at constant bottom-hole pressure (follows the code).** Both the code and eq. 5-91
   use `PID * lam_w * (P - Pbh)`. Eq. 5-64, an intermediate derivation step, uses the combined mobility
   as for injectors. That form is only right for an injector.
4. **The factor alpha (follows the code).** The code has `alpha = -Cswo / Csww`, which is what eliminates
   Sw when the two equations are added. Eq. 5-59 as printed has it inverted.
5. **Which coefficient goes with which neighbour (follows the code).** In the code `A` is built from the
   *minus* x transmissibility and multiplies `P(i-1)`; `C` is built from the *plus* x one and multiplies
   `P(i+1)`; the same holds for `E/F` (y) and `G/H` (z). Eq. 5-60 pairs them this way; the list of
   definitions in eq. 5-61 has the labels swapped. Python does not use these letters: it keeps one
   coefficient per direction (`coeff["x_plus"]`, ...), which removes the ambiguity.
6. **Gravity terms in the pressure equation (follows the code).** They are missing from the printed
   eq. 5-58 and 5-63 but present in the code (`GRA`) and in eq. 5-56 and 5-57.
7. **`C_pow` uses 1/Bw (follows the code).** Eq. 5-52 prints `c_r/B_o` inside the water bracket.
8. **Equilibrium initialization (follows the thesis and the code).** `Pcow = 0.433 (gamma_w - gamma_o) (WOC - depth)`
   and `P = P_datum + 0.433 gamma_o depth`. In `legacy/fortran-kusim` the initial capillary pressure is
   hard-coded to 32; the first version is used here.
9. **Minimum oil rate (follows the code, tidied).** The FORTRAN jumps to the end of the program (and so skips the
   saturation update of the last step). Python completes the step and then stops, so the saved
   pressure and saturation fields are consistent with the last well rates.
10. **Well control is per well (generalised).** The FORTRAN has one global production flag `PBC` and one
    injection flag `IBC`. Python gives each well its own `mode`. `read_legacy_input` copies the global
    flag to every well, so legacy files behave the same.
11. **`PCMULT` is applied (generalised).** The FORTRAN reads `PCMULT` but never uses it. Python multiplies
    the capillary-pressure column by it (1.0 in all the provided files).
12. **Solver (changed).** The default is a sparse direct solver instead of dense Gaussian elimination, which
    needs N² memory and N³ time. The textbook Gaussian elimination is kept as `linear_solver="gauss"`
    for teaching, and a test checks that both give the same pressures.
13. **Array layout (changed).** The FORTRAN fills the unused loop `I = (Z-1)*U + ...` and patches the
    boundaries with loops that use `NX` where they should use `NZ`. Python builds neighbour tables and a
    "closed face" mask once (`Grid`). Closed faces get zero transmissibility, which is the same no-flow boundary.
14. **Time loop (tidied).** `DO 123 T=0,TMAX,DT` with a real counter becomes `round(t_end/dt)` integer
    steps.
15. **Single precision constants.** The FORTRAN wrote constants such as `0.433` as single precision, which
    makes its results differ from a pure double-precision calculation in the 8th-9th digit. The tests allow for that.

16. **Wells completed in several layers (new).** The FORTRAN has one block per well. In Python a well takes a
    list of blocks (`Producer(block=grid.column_blocks(i, j), ...)`), each with its own Peaceman well index.
    A constant-pressure well draws from each perforation against its own bottom-hole pressure, which is
    solved implicitly like the single-block case. A rate-controlled well shares its rate between the
    perforations **in proportion to well index x mobility at the start of the step**, an explicit allocation
    of the same kind as the thesis's own oil/water split (eq. 5-79). The bottom-hole pressure then
    follows from the well equation summed over the perforations. This allocation ignores the pressure
    differences between layers, so a rate-controlled well cannot cross-flow; constant-pressure wells can.
17. **Wellbore pressure head (new).** The pressure in the wellbore increases with depth by the weight of the
    fluid in it. Without that, deeper perforations (whose blocks are naturally at higher pressure)
    back-flow into the well. A well's bottom-hole pressure is defined at its shallowest perforation, and
    the wellbore fluid gravity defaults to oil for producers and water for injectors (`wellbore_gravity`
    overrides it; 0 switches the head off). Single-block wells are unaffected. A constant fluid gradient is
    used, whatever the water cut.

18. **Upstream block chosen by potential, not by pressure (changed, 25 Oct 2026).** Thesis eq. 5-28 to 5-33 and
    the FORTRAN (`DO 31`, `IF (PO(I+1).GE.PO(I))`) take the mobility of a phase from the neighbour with the
    **higher phase pressure**. That is only right without gravity. The flow into a block is proportional to
    `(P_nb - P) - gamma * 0.433 * (z_nb - z)`, so the upstream block is the one with the higher *potential*,
    and Python now uses the sign of that expression (`Simulator._neighbour_is_upstream`). In vertical flow
    the pressure difference and the gravity head are of the same size and nearly cancel (about 14 to 18 psi
    over 40 ft), so the pressure test often picks the wrong block: sinking water was given the water
    mobility of the (water-poor) block below it and could not move down, which suppressed gravity
    segregation. The same applies to every connection of a dipping model. With equal depths the two rules
    are identical, so 1-D horizontal results (Buckley-Leverett) do not change, and neither do models
    without vertical flow (`kz = 0`).
    *Evidence.* (a) A closed column with water over oil must turn over: with the pressure rule the water stays on
    top, with the potential rule it sinks (`tests/test_upstream.py`). (b) Against ECLIPSE 100 on the same data (two
    cases, a 5x5x3 and the 6x4x3 bonus-project model, both with a producer and an injector in all layers) the RMS error of the
    producer bottom-hole pressure fell by a factor of 13 and 8, and the water breakthrough moved from 40 to 70
    days early to within 5 to 15 days (`eclipse_comparison/findings.md`). Changing the time step or refining
    the relative-permeability table did not help before the change. (c) The thesis's own Application 1
    (`examples/02_legacy_case.py`) changes by 0.01 % in cumulative oil: it has little vertical flow.
    *Consequence.* Results of layered models with vertical flow change after water breakthrough, for example the
    expected numbers of `course_plan/bonus_project.md` (regenerated). The thesis rule is kept as
    `Simulator(model, upstream="pressure")`, for reproducing the legacy behaviour and for teaching
    (`course/bonus/bonus_gravity_upstream.ipynb`). The decision is one small method that can be overridden in a subclass.
    Single-point upstream weighting by potential is the usual first-order scheme; it is not perfect for counter-current
    flow (gravity against viscous forces), which is not addressed here.

## What was checked

- First-step producer rates for `INPUT.DAT` match the old `QWP.OUT`/`QLP.OUT` to 8 digits (only the first step is
  comparable; see the unit change above).
- For multi-layer wells: rate shares follow well index x mobility; the perforation rates add up to the well
  total; the bottom-hole pressure satisfies the summed well equation; the material balance closes; a
  constant-pressure producer in hydrostatic equilibrium has the same drawdown in every layer; and the demo
  case has no back-flow (`tests/test_multilayer_wells.py`).
- Capillary-gravity equilibrium with no wells stays stationary (`test_equilibrium_start_is_stationary`).
- Material balance closes, and the error shrinks in proportion to `dt` (`test_material_balance_*`).
- The Gaussian-elimination and sparse solvers agree.
- Upstream by potential (decision 18): a water-over-oil column turns over (and stays put with the pressure rule),
  both rules agree for horizontal flow and for `kz = 0`, the rule can be overridden in a subclass, and
  `upstream="pressure"` reproduces the numbers of the earlier code exactly (`tests/test_upstream.py`; the
  expected table of the bonus project was regenerated with both rules).
- The oil and water balance of every step, added up, is 0.005 % and 0.014 % of the initial volumes for the bonus-project
  base case with the potential rule (0.05 % and 0.20 % with the pressure rule), and better than 0.001 % for a depletion run above the BHP floor. Compare the in-place volumes at the
  start and end of the run only with care: with the pore volume referred to the initial pressure that is path-dependent and
  overstates the error (`eclipse_comparison/findings.md`).
- A 1-D waterflood reproduces the Buckley-Leverett breakthrough time within 15 % and the saturation
  profile on average within 0.04 (`tests/test_buckley_leverett.py`, `examples/01_waterflood_1d.py`).

## Not done / open

- No numerical comparison with the old `*.OUT` files after step 1: the shipped executable does not match the source
  and could not be run (see `PLAN.md`), and decision 1 changes the well strength anyway.
- `RDATUM` is read but unused, as in the original.
- PVT interpolation differs from ECLIPSE: its `PVDO` interpolates `1/B` and `1/(B*mu)` linearly (per the comparison;
  the ECLIPSE manual was not consulted), `kusim` interpolates `1/B` and `mu`. Resampling the table with the ECLIPSE
  rule cut the remaining difference by a further 4 to 8 times. Not changed (`eclipse_comparison/findings.md`).
- Outside the PVT table `Table.lookup` is flat but `Table.slope` keeps the slope of the end segment (as in the
  original). The storage term then assumes fluid is released that the properties never account for, so mass is not
  conserved once the pressure leaves the table (about 1.5 % of the oil in place after 600 days in a test). Keep the
  producer's `min_bhp` above the bubble point. Not changed.
- Only one PVT region and the Cartesian block grid of the original; `dx` per column, `dy` per row, `dz` per layer.
