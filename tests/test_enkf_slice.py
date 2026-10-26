"""The history-matching slice used end to end: a reference EnKF (examples/) on a twin experiment.

These tests check the API (edits, restarts, ensembles, observations) by writing a complete Ensemble Kalman Filter
loop with it. They do not claim anything about the quality of the filter itself.
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow          # several seconds each: skip with  py -m pytest -m "not slow"

from conftest import REPO

sys.path.insert(0, str(REPO / "examples"))
from enkf_reference import correlated_log_field, enkf_analysis, gaspari_cohn      # noqa: E402
from enkf_twin import localisation_yd, make_model, run_twin_experiment           # noqa: E402


# ------------------------------------------------------------------ the reference analysis step
def test_the_analysis_matches_the_kalman_filter_for_a_linear_gaussian_problem():
    """State y ~ N(0, 1), data d = y + noise (sigma 1): the posterior is N(d_obs / 2, 1 / 2)."""
    rng = np.random.default_rng(0)
    ne = 40000
    Y = rng.standard_normal((1, ne))
    updated = enkf_analysis(Y, Y.copy(), np.array([2.0]), np.array([1.0]), rng)
    assert updated.mean() == pytest.approx(1.0, abs=0.03)
    assert updated.var(ddof=1) == pytest.approx(0.5, abs=0.03)


def test_localisation_with_ones_changes_nothing_and_with_zeros_blocks_the_update():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((5, 30))
    D = Y[:2] + 0.1 * rng.standard_normal((2, 30))
    args = (Y, D, np.array([0.5, -0.5]), np.array([0.2, 0.2]))
    plain = enkf_analysis(*args, np.random.default_rng(5))
    ones = enkf_analysis(*args, np.random.default_rng(5), rho_yd=np.ones((5, 2)), rho_dd=np.ones((2, 2)))
    zeros = enkf_analysis(*args, np.random.default_rng(5), rho_yd=np.zeros((5, 2)))
    assert np.allclose(plain, ones) and np.array_equal(zeros, Y)
    assert not np.allclose(plain, Y)


def test_gaspari_cohn_is_a_compact_correlation_function():
    d = np.linspace(0.0, 3.0, 301)
    rho = gaspari_cohn(d, 1.0)
    assert rho[0] == 1.0 and np.all(np.diff(rho) <= 1e-12) and np.all(rho >= 0)
    assert rho[d > 2.0].max() == 0.0 and gaspari_cohn(np.array([2.0]), 1.0)[0] == pytest.approx(0.0, abs=1e-12)
    assert gaspari_cohn(np.array([1.0 - 1e-9]), 1.0)[0] == pytest.approx(gaspari_cohn(np.array([1.0 + 1e-9]), 1.0)[0], abs=1e-6)
    assert gaspari_cohn(np.array([100.0]), 50.0)[0] == pytest.approx(gaspari_cohn(np.array([2.0]), 1.0)[0])


def test_the_random_fields_have_the_requested_statistics():
    grid = make_model(100.0).grid
    fields = correlated_log_field(grid, [120.0, 600.0, 250.0], 0.7, 1200.0, 60.0, np.random.default_rng(2), n=4000)
    assert fields.shape == (grid.n_cells, 4000)
    assert np.allclose(fields.mean(axis=1).reshape(3, -1).mean(axis=1), np.log([120.0, 600.0, 250.0]), atol=0.06)
    assert fields.std(axis=1).mean() == pytest.approx(0.7, abs=0.04)
    near = np.corrcoef(fields[0], fields[1])[0, 1]                 # neighbouring blocks, 400 ft apart
    far = np.corrcoef(fields[0], fields[5])[0, 1]                  # 2000 ft apart
    assert near > far and near > 0.5


# ------------------------------------------------------------------ the twin experiment
@pytest.fixture(scope="module")
def twin():
    return run_twin_experiment(ne=30, seed=0)


def test_the_twin_experiment_matches_the_data_after_the_updates(twin):
    m = twin["metrics"]
    assert m["posterior_member_rms"] < 0.2 * m["prior_member_rms"]        # tens of sigma before, a few sigma after
    assert m["posterior_member_rms"] < 5.0
    assert m["posterior_ok"] == m["members"]                              # no diagnostics warnings after the updates


def test_the_twin_experiment_moves_the_permeability_towards_the_truth_and_narrows_the_ensemble(twin):
    m = twin["metrics"]
    assert m["posterior_logk_rmse"] < 0.8 * m["prior_logk_rmse"]
    assert m["posterior_spread"] < 0.8 * m["prior_spread"]


def test_every_member_keeps_the_log_of_its_edits(twin):
    model = twin["posterior_models"][0]
    assert len(model.provenance) == len(twin["dates"])
    assert model.provenance[0] == "EnKF update at day 50" and model.provenance[-1] == "EnKF update at day 600"
    assert twin["prior_models"][0].provenance == []                         # the prior models were not touched


def test_the_updated_models_stay_physical(twin):
    for model in twin["posterior_models"]:
        assert np.all(model.kx >= 1.0) and np.all(model.kx <= 1e4)
        assert np.allclose(model.ky, 0.8 * model.kx) and np.allclose(model.kz, 0.1 * model.kx)   # the ties held


def test_an_invalid_truth_is_rejected():
    with pytest.raises(ValueError, match="leaves the PVT table"):
        run_twin_experiment(ne=3, seed=1, std=2.0)        # very heterogeneous: the pressure reaches 10 000 psia


def test_the_truth_and_the_prior_members_do_not_depend_on_the_ensemble_size():
    small = run_twin_experiment(ne=3, seed=0, t_end=100.0)
    large = run_twin_experiment(ne=5, seed=0, t_end=100.0)
    assert np.array_equal(small["truth_model"].kx, large["truth_model"].kx)
    assert small["observations"].table.equals(large["observations"].table)
    for a, b in zip(small["prior_models"], large["prior_models"][:3]):
        assert np.array_equal(a.kx, b.kx)


def test_localisation_keeps_blocks_far_from_the_wells_at_their_prior_values():
    out = run_twin_experiment(ne=30, seed=0, localise_length=500.0)
    model = out["truth_model"]
    x, y, _ = model.grid.cell_centres()
    wells = [model.grid.cell_index(w.blocks[0]) for w in [*model.producers, *model.injectors]]
    far = np.all([np.hypot(x - x[c], y - y[c]) > 1000.0 for c in wells], axis=0)       # beyond 2 L from both wells
    assert far.sum() >= 6
    prior, post = out["params"].matrix(out["prior_models"]), out["params"].matrix(out["posterior_models"])
    assert np.allclose(post[far], prior[far])                               # untouched
    assert not np.allclose(post[~far], prior[~far])                         # while the blocks near the wells moved
    assert out["metrics"]["posterior_member_rms"] < out["metrics"]["prior_member_rms"]


def test_localisation_matrix_has_the_shape_the_analysis_needs():
    model = make_model(100.0)
    rho = localisation_yd(model, ["WBHP:P1", "WBHP:I1", "WWCT:P1"], 800.0)
    assert rho.shape == (3 * model.grid.n_cells, 3)
    assert rho.max() == 1.0 and rho.min() == 0.0
    assert np.array_equal(rho[:, 0], rho[:, 2])                              # both data belong to the producer
    assert not np.array_equal(rho[:, 0], rho[:, 1])


def test_members_can_be_run_on_several_processes_in_the_filter():
    serial = run_twin_experiment(ne=6, seed=0, t_end=200.0, first_date=50.0, interval=50.0)
    parallel = run_twin_experiment(ne=6, seed=0, t_end=200.0, first_date=50.0, interval=50.0, workers=2)
    for a, b in zip(serial["posterior_models"], parallel["posterior_models"]):
        assert np.array_equal(a.kx, b.kx)
