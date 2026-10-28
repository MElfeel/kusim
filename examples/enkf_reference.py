"""A compact reference Ensemble Kalman Filter analysis step, to exercise the kusim history-matching API.

The code now lives in ``kusim.ensemble`` (it is what ``kusim ensemble`` runs); this module re-exports it so that the
examples and notebooks keep working.  It was written to show that ``kusim.parameters``, ``kusim.history`` and
``Simulator.set_state`` are enough to write a filter, and follows
the stochastic EnKF of ``enkf/HA_localised_enkf.docx``:

    y_j^a = y_j^f + rho_YD o C_YD (rho_DD o C_DD + C_e)^-1 (d_obs,j - d_j^f)

where ``y`` is the augmented state (parameters and dynamic variables), ``d`` the simulated data, ``d_obs,j`` the
observations perturbed with the measurement error, ``C`` ensemble covariances and ``rho`` an optional
localisation (correlation) matrix applied with the Schur (element-wise) product ``o``.
"""

from kusim.ensemble import correlated_log_field, enkf_analysis, gaspari_cohn, log_field_covariance  # noqa: F401
