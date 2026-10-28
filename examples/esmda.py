"""Ensemble Smoother with Multiple Data Assimilation (ES-MDA) on the kusim history-matching API.

The code lives in ``kusim.ensemble`` (``kusim ensemble --method esmda`` runs it); this module re-exports it for the examples and notebooks.
The idea in plain words: run every model over the *whole* history, compare with *all* the data, and improve the uncertain
parameters (here the permeability) a little.  Then run again from day 0 and repeat.  Each pass pretends that the data are
noisier than they are, by a factor ``alpha`` on the variance, so that each step is small and careful.  The factors satisfy

    sum over the passes of 1 / alpha  =  1,

so that together the passes use the data exactly once.  The permeability does not change in reality; only our knowledge of
it does, and the models are always re-run from time zero, so they are consistent with the parameters by construction.

The update of one pass is the stochastic EnKF analysis of ``enkf_reference.enkf_analysis`` with the measurement errors
multiplied by ``sqrt(alpha)``, applied to the parameters only (the data of all dates stacked into one vector).
"""

from kusim.ensemble import check_alphas, enkf_analysis, esmda_alphas, esmda_update, stack_data  # noqa: F401
