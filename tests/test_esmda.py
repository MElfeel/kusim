"""ES-MDA (examples/esmda.py, examples/esmda_twin.py) used end to end on the history-matching API.

Like ``test_enkf_slice.py`` these tests check the workflow and the update rule, and claim little about the quality of the method.
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow          # several seconds each: skip with  py -m pytest -m "not slow"

from conftest import REPO

sys.path.insert(0, str(REPO / "examples"))
from esmda import check_alphas, esmda_alphas, esmda_update      # noqa: E402
from esmda_twin import run_esmda_experiment                     # noqa: E402


# ------------------------------------------------------------------ the update rule
def test_equal_alphas_use_the_data_exactly_once():
    for n in (1, 2, 4, 7):
        alphas = esmda_alphas(n)
        assert len(alphas) == n and sum(1.0 / a for a in alphas) == pytest.approx(1.0)
        check_alphas(alphas)
    check_alphas([2.0, 4.0, 4.0])                                   # unequal factors are fine if they satisfy the rule
    with pytest.raises(ValueError, match="sum"):
        check_alphas([2.0, 2.0, 2.0])


def test_several_passes_reproduce_the_kalman_filter_for_a_linear_gaussian_problem():
    """State y ~ N(0, 1), data d = y + noise (sigma 1): the posterior is N(d_obs / 2, 1 / 2), after one pass or four."""
    ne = 40000
    for n_passes in (1, 4):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((1, ne))
        for alpha in esmda_alphas(n_passes):
            Y = esmda_update(Y, Y.copy(), np.array([2.0]), np.array([1.0]), alpha, rng)     # linear model: the data are the state
        assert Y.mean() == pytest.approx(1.0, abs=0.03)
        assert Y.var(ddof=1) == pytest.approx(0.5, abs=0.03)


def test_an_update_with_a_huge_alpha_hardly_moves_the_parameters():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((3, 50))
    D = Y[:2] + 0.1 * rng.standard_normal((2, 50))
    small = esmda_update(Y, D, np.array([1.0, -1.0]), np.array([0.2, 0.2]), 1.0, np.random.default_rng(5))
    huge = esmda_update(Y, D, np.array([1.0, -1.0]), np.array([0.2, 0.2]), 1e6, np.random.default_rng(5))
    assert np.abs(huge - Y).max() < 0.05 * np.abs(small - Y).max()


# ------------------------------------------------------------------ the twin experiment
@pytest.fixture(scope="module")
def twin():
    return run_esmda_experiment(ne=30, seed=0, n_passes=4)


def test_the_twin_experiment_matches_the_data_after_the_passes(twin):
    m = twin["metrics"]
    assert len(m["rms_by_pass"]) == 5 and m["rms_by_pass"][0] == m["prior_member_rms"]
    assert m["posterior_member_rms"] < 0.2 * m["prior_member_rms"]
    assert m["posterior_member_rms"] < 5.0
    assert m["posterior_ok"] == m["members"]


def test_the_twin_experiment_narrows_the_ensemble(twin):
    m = twin["metrics"]
    assert m["posterior_spread"] < 0.8 * m["prior_spread"]


def test_every_member_keeps_the_log_of_its_edits(twin):
    model = twin["posterior_models"][0]
    assert model.provenance == [f"ES-MDA pass {n} of 4 (alpha 4)" for n in (1, 2, 3, 4)]
    assert twin["prior_models"][0].provenance == []                         # the prior models were not touched


def test_the_updated_models_stay_physical(twin):
    for model in twin["posterior_models"]:
        assert np.all(model.kx >= 1.0) and np.all(model.kx <= 1e4)
        assert np.allclose(model.ky, 0.8 * model.kx) and np.allclose(model.kz, 0.1 * model.kx)


def test_the_truth_and_prior_are_those_of_the_enkf_twin():
    from enkf_twin import make_twin
    out = run_esmda_experiment(ne=3, seed=0, n_passes=1, t_end=100.0)
    twin = make_twin(3, 0, t_end=100.0)
    assert np.array_equal(out["truth_model"].kx, twin["truth_model"].kx)
    assert out["observations"].table.equals(twin["observations"].table)


def test_bad_alphas_are_rejected():
    with pytest.raises(ValueError, match="sum"):
        run_esmda_experiment(ne=3, seed=0, alphas=[2.0, 2.0, 2.0], t_end=100.0)

