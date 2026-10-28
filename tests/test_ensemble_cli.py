"""kusim ensemble: ES-MDA and EnKF from the command line, and the prior they start from."""

import json

import numpy as np
import pytest

from kusim.cases import layered_waterflood
from kusim.cli import main
from kusim.ensemble import (check_alphas, check_prior, enkf_analysis, esmda_alphas, gaspari_cohn, localisation_yd, run_esmda,
                            sample_prior)
from kusim.history import Observations, run_to
from kusim.parameters import FieldParameter, ParameterSet, ScalarParameter
from kusim.schema import get_schema
from kusim.spec import SpecError, load_model, prior_from_dict, save_model, save_observations

TIMES = [10.0, 20.0, 30.0]
PARAMS = {"parameters": [{"type": "field", "field": "kx", "ties": ["ky", "kz"], "lower": 1.0}]}
PRIOR = {"parameters": {"log_kx": {"std": 0.6, "range_xy": 150.0, "range_z": 20.0}}}


def _cli(capsys, *argv):
    code = main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """A small twin: the base model, a truth with other permeability, and noisy-free data of the truth."""
    tmp = tmp_path_factory.mktemp("ensemble")
    model = layered_waterflood(nx=4, ny=3, nz=2, days=30.0)
    save_model(model, tmp / "m.json")
    truth = model.with_changes(kx=2.0 * model.kx, ky=2.0 * model.ky, kz=2.0 * model.kz)
    table = run_to(truth, TIMES).table
    rows = [{"time": float(r.TIME), "vector": v, "value": float(getattr(r, v.replace(":", "_"))), "sigma": s}
            for r in table.rename(columns=lambda c: c.replace(":", "_")).itertuples()
            for v, s in (("WBHP:I1", 5.0), ("WWCT:P1", 0.02), ("WOPR:P1", 20.0))]
    save_observations(Observations(rows), tmp / "o.json")
    (tmp / "p.json").write_text(json.dumps(PARAMS))
    (tmp / "prior.json").write_text(json.dumps(PRIOR))
    return model, tmp


def _args(tmp, method, *more):
    return ("ensemble", tmp / "m.json", "--method", method, "--observations", tmp / "o.json", "--parameters",
            tmp / "p.json", "--prior", tmp / "prior.json", "--members", 12, *more)


# ---------------------------------------------------------------------- the analysis step
def test_alphas():
    assert esmda_alphas(4) == [4.0] * 4
    check_alphas([2.0, 2.0])
    with pytest.raises(ValueError, match="sum"):
        check_alphas([2.0, 3.0])


def test_enkf_analysis_moves_the_ensemble_toward_the_data():
    rng = np.random.default_rng(0)
    Y = rng.normal(0.0, 1.0, (1, 200))
    D = 2.0 * Y                                                   # the data are twice the parameter
    out = enkf_analysis(Y, D, np.array([2.0]), np.array([0.1]), rng)       # observed 2.0: the parameter is about 1
    assert abs(out.mean() - 1.0) < 0.1 and out.std() < Y.std()


def test_gaspari_cohn():
    d = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(gaspari_cohn(d, 1.0), [1.0, 5.0 / 24.0, 0.0, 0.0], atol=1e-12)
    assert gaspari_cohn(0.0, 100.0) == 1.0 and gaspari_cohn(250.0, 100.0) < gaspari_cohn(50.0, 100.0)


def test_localisation_leaves_field_data_alone():
    model = layered_waterflood(nx=4, ny=3, nz=2, days=5.0)
    rho = localisation_yd(model, ["WBHP:P1", "FOPT"], 300.0, n_blocks=3)
    assert rho.shape == (3 * model.grid.n_cells, 2) and np.all(rho[:, 1] == 1.0)


# ---------------------------------------------------------------------- the prior
@pytest.fixture
def small():
    model = layered_waterflood(nx=4, ny=3, nz=2, days=5.0)
    return model, ParameterSet([FieldParameter("kx", ties=("ky", "kz"), lower=1.0)])


def test_prior_is_reproducible_and_independent_of_the_size(small):
    model, params = small
    a, _ = sample_prior(model, params, PRIOR["parameters"], 6, seed=3)
    b, _ = sample_prior(model, params, PRIOR["parameters"], 6, seed=3)
    c, _ = sample_prior(model, params, PRIOR["parameters"], 10, seed=3)
    d, _ = sample_prior(model, params, PRIOR["parameters"], 6, seed=4)
    assert all(np.array_equal(x.kx, y.kx) for x, y in zip(a, b))
    assert all(np.array_equal(x.kx, y.kx) for x, y in zip(a, c))            # member j is member j in a larger ensemble
    assert not np.array_equal(a[0].kx, d[0].kx)


def test_prior_statistics_and_correlation(small):
    model, params = small
    prior = {"log_kx": {"std": 0.5, "range_xy": 400.0, "range_z": 40.0}}
    models, matrix = sample_prior(model, params, prior, 400, seed=1)
    base = np.log(model.kx)
    assert np.abs(matrix.mean(axis=1) - base).max() < 0.15                  # the mean is the model's value
    assert np.allclose(matrix.std(axis=1, ddof=1), 0.5, atol=0.1)
    independent = sample_prior(model, params, {"log_kx": {"std": 0.5}}, 400, seed=1)[1]
    near = lambda m: np.corrcoef(m[0] - base[0], m[1] - base[1])[0, 1]      # neighbouring blocks in x
    assert near(matrix) > 0.4 > abs(near(independent))


def test_prior_mean_by_layer_and_scalar_parameters(small):
    model, _ = small
    params = ParameterSet([FieldParameter("kx", ties=("ky", "kz")), ScalarParameter("rock_compressibility", "log")])
    prior = {"log_kx": {"std": 1e-6, "mean_by_layer": [50.0, 400.0]},
             "log_rock_compressibility": {"std": 1e-6, "mean": 3e-6}}
    models, _ = sample_prior(model, params, prior, 3, seed=0)
    by_layer = lambda m: [m.kx[model.grid.k == k][0] for k in (0, 1)]
    assert np.allclose(by_layer(models[0]), [50.0, 400.0], rtol=1e-4)
    assert models[0].rock_compressibility == pytest.approx(3e-6, rel=1e-4)


def test_prior_errors(small):
    model, params = small
    with pytest.raises(ValueError, match="exactly one entry"):
        check_prior({"log_ky": {"std": 1.0}}, params)
    with pytest.raises(ValueError, match="std"):
        check_prior({"log_kx": {"std": -1.0}}, params)
    with pytest.raises(ValueError, match="not both"):
        check_prior({"log_kx": {"std": 1.0, "mean": 1.0, "mean_by_layer": [1, 2]}}, params)
    with pytest.raises(ValueError, match="one per layer"):
        sample_prior(model, params, {"log_kx": {"std": 1.0, "mean_by_layer": [1.0, 2.0, 3.0]}}, 3)
    with pytest.raises(ValueError, match="at least 2"):
        sample_prior(model, params, {"log_kx": {"std": 1.0}}, 1)
    big = layered_waterflood(nx=60, ny=60, nz=1, days=5.0)
    with pytest.raises(ValueError, match="limit"):
        sample_prior(big, ParameterSet([FieldParameter("kx")]), {"log_kx": {"std": 1.0, "range_xy": 100.0}}, 3)


def test_prior_json_errors_are_located_and_the_schema_accepts_it():
    assert prior_from_dict(PRIOR)["log_kx"]["std"] == 0.6
    with pytest.raises(SpecError) as error:
        prior_from_dict({"parameters": {"log_kx": {"stdev": 1.0}}})
    assert error.value.path == "parameters.log_kx.stdev" and error.value.details["did_you_mean"] == {"stdev": "std"}
    with pytest.raises(SpecError) as error:
        prior_from_dict({"parameters": {"log_kx": {"range_xy": 5.0}}})
    assert (error.value.code, error.value.path) == ("missing_key", "parameters.log_kx.std")
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(PRIOR, get_schema("prior"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"parameters": {"log_kx": {"std": 0}}}, get_schema("prior"))


# ---------------------------------------------------------------------- the command
def test_esmda_improves_the_fit(case, capsys):
    _, tmp = case
    code, out = _cli(capsys, *_args(tmp, "esmda", "--out", tmp / "out_esmda", "--write-members", "--write-ensemble"))
    m = out["metrics"]
    assert code == 0 and out["ok"] and out["healthy"] and m["method"] == "esmda" and m["members"] == 12
    assert m["posterior_member_rms"] < 0.5 * m["prior_member_rms"] and m["posterior_spread"] < m["prior_spread"]
    assert m["rms_by_pass"][0] == m["prior_member_rms"] and m["rms_by_pass"][-1] == m["posterior_member_rms"]
    assert len(m["rms_by_pass"]) == 5 and m["alphas"] == [4.0] * 4 and 0 <= m["best_member"] < 12
    assert set(m["posterior_rms_by_vector"]) == {"WBHP:I1", "WOPR:P1", "WWCT:P1"}
    folder = tmp / "out_esmda"
    mean_model = load_model(folder / "posterior_mean.json")
    assert mean_model.grid.n_cells == 24
    assert len(list((folder / "members").glob("*.json"))) == 12
    parameters = json.loads((folder / "ensemble_parameters.json").read_text())
    assert np.array(parameters["posterior"]).shape == (12, 24) and parameters["names"] == {"log_kx": [0, 24]}
    assert json.loads((folder / "ensemble.json").read_text())["metrics"] == out["metrics"]


def test_posterior_mean_model_fits_the_data(case, capsys):
    _, tmp = case
    _, out = _cli(capsys, *_args(tmp, "esmda", "--out", tmp / "out_mean"))
    code, check = _cli(capsys, "misfit", tmp / "out_mean" / "posterior_mean.json", "--observations", tmp / "o.json")
    assert check["rms"] == pytest.approx(out["metrics"]["posterior_mean_rms"], rel=1e-9)
    _, prior_check = _cli(capsys, "misfit", tmp / "m.json", "--observations", tmp / "o.json")
    assert check["rms"] < prior_check["rms"]


def test_the_same_seed_gives_the_same_result(case, capsys):
    _, tmp = case
    _, a = _cli(capsys, *_args(tmp, "esmda", "--seed", 5))
    _, b = _cli(capsys, *_args(tmp, "esmda", "--seed", 5))
    _, c = _cli(capsys, *_args(tmp, "esmda", "--seed", 6))
    assert a["metrics"] == b["metrics"] and a["metrics"] != c["metrics"]


def test_custom_alphas(case, capsys):
    _, tmp = case
    _, out = _cli(capsys, *_args(tmp, "esmda", "--alphas", "2,2"))
    assert out["metrics"]["alphas"] == [2.0, 2.0] and len(out["metrics"]["rms_by_pass"]) == 3
    code, out = _cli(capsys, *_args(tmp, "esmda", "--alphas", "2,3"))
    assert code == 2 and "sum" in out["error"]


def test_enkf_improves_the_fit(case, capsys):
    _, tmp = case
    code, out = _cli(capsys, *_args(tmp, "enkf", "--localise", 300))
    m = out["metrics"]
    assert code == 0 and out["ok"] and m["method"] == "enkf" and m["localise_length"] == 300.0
    assert m["posterior_member_rms"] < m["prior_member_rms"] and "saturations_clipped" in m


def test_workers_do_not_change_the_result(case, capsys):
    _, tmp = case
    _, one = _cli(capsys, *_args(tmp, "esmda", "--passes", 2))
    _, two = _cli(capsys, *_args(tmp, "esmda", "--passes", 2, "--workers", 2))
    assert one["metrics"] == two["metrics"]


def test_few_members_are_warned_about(case, capsys):
    _, tmp = case
    (tmp / "narrow.json").write_text(json.dumps({"parameters": {"log_kx": {"std": 0.01, "mean": 1000.0}}}))
    code, out = _cli(capsys, "ensemble", tmp / "m.json", "--method", "esmda", "--observations", tmp / "o.json",
                     "--parameters", tmp / "p.json", "--prior", tmp / "narrow.json", "--members", 8)
    assert code == 0 and out["metrics"]["members"] == 8
    assert any("fewer than 10" in w for w in out["warnings"])


def test_bad_arguments_and_prior_mismatch(case, capsys):
    _, tmp = case
    code, out = _cli(capsys, *_args(tmp, "enkf", "--passes", 3))
    assert code == 2 and out["code"] == "bad_argument"
    code, out = _cli(capsys, *_args(tmp, "esmda", "--localise", 300))
    assert code == 2 and out["code"] == "bad_argument"
    code, out = _cli(capsys, "ensemble", tmp / "m.json", "--method", "esmda", "--observations", tmp / "o.json",
                     "--parameters", tmp / "p.json", "--prior", tmp / "prior.json", "--members", 1)
    assert code == 2 and out["code"] == "bad_argument"
    (tmp / "wrong.json").write_text(json.dumps({"parameters": {"log_ky": {"std": 0.5}}}))
    code, out = _cli(capsys, "ensemble", tmp / "m.json", "--method", "esmda", "--observations", tmp / "o.json",
                     "--parameters", tmp / "p.json", "--prior", tmp / "wrong.json")
    assert code == 2 and out["code"] == "prior_mismatch" and out["allowed"] == ["log_kx"]
    (tmp / "typo.json").write_text(json.dumps({"parameters": {"log_kx": {"sd": 0.5}}}))
    code, out = _cli(capsys, "ensemble", tmp / "m.json", "--method", "esmda", "--observations", tmp / "o.json",
                     "--parameters", tmp / "p.json", "--prior", tmp / "typo.json")
    assert code == 2 and out["code"] == "unknown_key" and out["path"] == "parameters.log_kx.sd"
    with pytest.raises(SystemExit):
        main(["ensemble", str(tmp / "m.json")])
    capsys.readouterr()


def test_python_api_matches_the_command(case, capsys):
    model, tmp = case
    obs = Observations(json.loads((tmp / "o.json").read_text()))
    params = ParameterSet([FieldParameter("kx", ties=("ky", "kz"), lower=1.0)])
    result = run_esmda(model, params, PRIOR["parameters"], obs, members=12, seed=11)
    _, out = _cli(capsys, *_args(tmp, "esmda"))
    assert result["metrics"]["posterior_member_rms"] == out["metrics"]["posterior_member_rms"]
