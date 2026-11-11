# KUSIM vs ECLIPSE comparison (not part of the course)

`kusim_vs_eclipse.ipynb` builds a two-phase waterflood in kusim, writes an equivalent ECLIPSE deck, and compares the results.
Read `findings.md` first: it records what the comparison found (the upstream rule, now corrected in the package) and what did not matter.

1. Run the notebook from this folder (`py -m jupyter lab`, or open it in VS Code). Choose the case with `CASE_NAME`:
   `deck_5x5x3` (the case in `eclipse/`, with its ECLIPSE results) or `bonus_6x4x3` (the bonus-project model, results in `eclipse_runs/bonus_case/`).
2. The deck is written into the case folder only if there is none; an existing deck is compared with the case and never changed.
   To run another case, put its deck in its own folder and run `eclrun eclipse KUSIM_BONUS.DATA` there.
3. The ECLIPSE result is read from `KUSIM_BONUS.UNSMRY` + `.SMSPEC` (needs `pip install resdata`) or from `KUSIM_BONUS.csv`
   (time in days, columns `WBHP:P1`, `WOPR:P1`, `FPRP`, ...).
4. `investigation_upstream.py` reproduces the table of `findings.md` for the 5x5x3 case without the notebook.

Use `FPRP` (pore-volume average pressure), not `FPR`. Keep pressures inside the PVT table.
`Simulator(model, upstream="pressure")` is the thesis rule for the upstream block, kept for reproducing the old behaviour.
