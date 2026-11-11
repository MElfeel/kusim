# kusim vs ECLIPSE: findings

**Status (25 Oct 2026): resolved in the package.** The main cause of the mismatch was the upstream rule. `kusim` now chooses
the upstream block by potential (`docs/porting-notes.md`, decision 18); the thesis rule is still available as
`Simulator(model, upstream="pressure")`. The tables below are measured, the pressure-rule rows are what `kusim` gave before the change.

Reproduce: `py eclipse_comparison/investigation_upstream.py` (the 5x5x3 case), or run `kusim_vs_eclipse.ipynb` with
`CASE_NAME = "deck_5x5x3"` or `"bonus_6x4x3"` (Section 7 has the switches).

## Cause of the large mismatch: upstream weighting by pressure instead of potential

`simulator.py` (and the thesis eq. 5-28 to 5-33, and the FORTRAN) picked the upstream block by comparing phase **pressure**.
With gravity the upstream block must be chosen by **potential** (pressure minus gravity head). In the vertical direction the
gravity head dominates, so water in the high-permeability layer 2 (which should slump into layer 3) was weighted with layer 3's
near-zero krw. Gravity segregation was suppressed. A closed column with water on top of oil is the simplest test: the water
stays on top with the pressure rule and sinks with the potential rule (`tests/test_upstream.py`).

Evidence from the ECLIPSE comparison (5x5x3 case): block pressures agreed with ECLIPSE within 1-3 psi (layer means, up to
500 days) and the total water in the model agreed, but at 500 days the layer-average Sw (kusim with the pressure rule / ECLIPSE) was
layer 2: 0.56 / 0.48 and layer 3: 0.26 / 0.35.

## RMS difference over the whole run (psi, psi, psi, STB/d, fraction)

5x5x3 case (2000 days), ECLIPSE results in `eclipse/`:

| kusim variant | FPRP | WBHP P1 | WBHP I1 | WOPR P1 | WWCT P1 |
|---|---|---|---|---|---|
| thesis rule (upstream by pressure) | 787 | 908 | 789 | 1564 | 0.156 |
| upstream by potential (the default now) | 100 | 70 | 107 | 200 | 0.020 |
| + ECLIPSE-style PVT interpolation | 19 | 8 | 23 | 51 | 0.005 |

Bonus-project 6x4x3 case (1200 days), ECLIPSE results in `eclipse_runs/bonus_case/` and `course/bonus/data/`:

| kusim variant | FPRP | WBHP P1 | WBHP I1 | WOPR P1 | WWCT P1 |
|---|---|---|---|---|---|
| thesis rule (upstream by pressure) | 282 | 325 | 290 | 812 | 0.101 |
| upstream by potential (the default now) | 48 | 42 | 58 | 116 | 0.015 |
| + ECLIPSE-style PVT interpolation | 13 | 29 | 24 | 40 | 0.005 |

Water breakthrough (water cut above 1 %): 5x5x3 case, ECLIPSE day 545; pressure rule 475, potential rule 530, potential rule with the
ECLIPSE-style PVT 555. Bonus case, ECLIPSE day 415; pressure rule 375, potential rule 410. Late field pressure (day 2000, 5x5x3 case)
differs by 0.5 % with the ECLIPSE-style PVT.

The thesis's own Application 1 (`examples/02_legacy_case.py`) changes by 0.01 % in cumulative oil: it has little vertical flow.

## Second cause: PVDO interpolation

ECLIPSE interpolates 1/B and 1/(B*mu) linearly; kusim interpolates 1/B and mu (the ECLIPSE manual was not consulted: the
hypothesis is supported by the match improving). Resampling the PVT table with the ECLIPSE rule (`eclipse_pvt=True` in the
notebook) removes most of what remains. Not changed in the package.

## Things that do NOT explain the mismatch

* **Time step**: after the upstream fix dt = 1, 2.5, 5, 10 all give the same quality (the explicit scheme's dispersion even
  compensates a little at dt = 10). Before the fix dt changed nothing because the upstream error dominated.
* **Finer relative-permeability table** (41 points, smooth PCHIP through the 6 points, same table in both codes, ECLIPSE re-run):
  the mismatch is unchanged (kusim and ECLIPSE both move, by the same amount; ECLIPSE alone changes by up to 68 psi and 0.02 in
  water cut). Both interpolate linearly, so there is no resolution gap. (The deck is `eclipse_runs/fine_swof/KUSIM_BONUS.DATA`.)
* **PVT range**: the pressures stay inside the table (extending PVDO above 6000 psia changes nothing).
* **Wellbore head**: the effect is mixed. Switching it off makes the producer BHP worse in the 5x5x3 case (8 to 20 psi RMS, with the ECLIPSE-style PVT)
  and the injector BHP worse (23 to 38 psi); in the bonus case it improves the producer BHP and worsens the injector BHP.

## Mass balance: use the per-step balance

The earlier version of the notebook compared the oil and water in place at the end of the run with the start, with the pore volume
referred to the initial pressure. That is path-dependent (rock compaction) and overstated the error: for the base case it gave 0.15 %
(oil) and 0.59 % (water) with the new rule, against 0.06 % and 0.23 % with the old one, which wrongly suggested that the change made
conservation worse. Adding up the balance of every step (pore volume referred to the start of the step, which is what the scheme
assumes) gives, for the bonus base case: **0.005 % oil and 0.014 % water with the potential rule, 0.05 % and 0.20 % with the pressure
rule**; a depletion run above the BHP floor closes to better than 0.001 %. Outside the PVT table the balance fails: about 1.5 % of the
oil in place is missing after 600 days (`Table.slope` keeps the end-segment slope while `Table.lookup` is flat; see
`docs/porting-notes.md`, "Not done / open").

## Housekeeping

* The ECLIPSE results in `eclipse/` belong to the 5x5x3 deck, not to the bonus-project data. The notebook now has both cases
  (`CASE_NAME`), each with its own folder, and compares an existing deck with the case instead of overwriting it.
* `TABDIMS` in the deck writer sizes NSSFUN to the saturation table (it was fixed at 20 rows).
* `resdata` (Equinor, GPL-3.0) is needed to read `.UNSMRY` directly; the notebook falls back to a CSV export.
* `course_plan/bonus_project.md` was regenerated with the new default, and `course/bonus/bonus_gravity_upstream.ipynb` teaches the change.
