"""The two adjoint workflows of examples/ (history matching and production optimisation), at reduced size."""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow          # several seconds each: skip with  py -m pytest -m "not slow"

from conftest import REPO

sys.path.insert(0, str(REPO / "examples"))
from adjoint_workflows import allocate_injection, history_match                   # noqa: E402


@pytest.fixture(scope="module")
def matched():
    return history_match(seed=0, maxiter=40)


def test_history_matching_with_adjoint_gradients_reaches_the_noise_level(matched):
    m = matched["metrics"]
    assert m["start_data_rms"] > 20.0 and m["end_data_rms"] < 2.0              # 30 sigma before, about 1 sigma after
    assert matched["history"][-1] < 1e-2 * matched["history"][0]
    assert matched["evaluations"] < 120


def test_history_matching_moves_the_permeability_towards_the_truth(matched):
    m = matched["metrics"]
    assert m["end_logk_rmse"] < 0.8 * m["start_logk_rmse"]
    assert np.abs(m["end_layers"] - m["truth_layers"]).max() < np.abs(m["start_layers"] - m["truth_layers"]).max()


def test_the_adjoint_gradient_of_the_matching_problem_agrees_with_finite_differences(matched):
    check = matched["check"]
    assert check.stable.mean() >= 0.75 and check.relative.max() < 1e-5


def test_a_gradient_costs_a_few_simulations_and_finite_differences_cost_two_per_parameter(matched):
    assert matched["seconds_per_gradient"] < 10 * matched["seconds_per_run"]            # about 3
    assert 2 * matched["problem"].n_params * matched["seconds_per_run"] > 10 * matched["seconds_per_gradient"]


def test_the_correlated_prior_is_what_makes_the_estimate_good(matched):
    plain = history_match(seed=0, maxiter=40, correlated_prior=False)["metrics"]
    assert matched["metrics"]["end_logk_rmse"] < plain["end_logk_rmse"]


@pytest.fixture(scope="module")
def allocated():
    return allocate_injection(compare_nelder_mead=False)


def test_the_optimised_injection_beats_the_uniform_one_when_the_simulator_is_run_again(allocated):
    assert allocated["profit_end"] > allocated["profit_start"]
    final = allocated["final"].table.iloc[0]
    assert allocated["profit_end"] == pytest.approx(60.0 * final["FOPT"] - 20.0 * final["FWPT"] - 3.0 * final["FWIT"], rel=1e-9)


def test_the_optimised_injection_obeys_the_bounds_and_the_volume(allocated):
    x = allocated["x"]
    assert x.sum() == pytest.approx(4 * 9000.0, abs=1e-3)
    assert np.all(x >= 2000.0 - 1e-6) and np.all(x <= 14000.0 + 1e-6)


def test_the_optimiser_never_left_the_physical_region(allocated):
    assert allocated["backflow_on_the_path"] == 0 and allocated["mode_switches_on_the_path"] == 0
    d = allocated["final"].diagnostics
    assert not d["outside_pvt_range"] and d["n_backflow_well_steps"] == 0


def test_the_adjoint_gradient_of_the_injection_problem_agrees_with_finite_differences(allocated):
    assert allocated["check"].stable.all() and allocated["check"].relative.max() < 1e-6


def test_the_optimum_agrees_with_a_derivative_free_search():
    both = allocate_injection(compare_nelder_mead=True)
    assert both["profit_end"] == pytest.approx(both["nelder_mead"]["profit"], rel=1e-4)
    assert both["evaluations"] * 5 < both["nelder_mead"]["evaluations"]            # far fewer evaluations
