"""The edit layer: ``ReservoirModel.with_changes`` and ``ParameterSet``."""

import numpy as np
import pytest

from kusim import Equilibrium, Injector, Producer
from kusim.parameters import FieldParameter, ParameterSet, ScalarParameter

from conftest import make_model


def _model():
    model = make_model(nx=4, ny=3, nz=2, k=200.0, dt=2.0, t_end=20.0,
                       initial=Equilibrium(datum_pressure=3000.0, woc_depth=200.0),
                       injectors=[Injector(1, "water_rate", 50.0)], producers=[Producer(24, "bhp", 2500.0)])
    model.ky = model.kx * 0.8
    model.kz = model.kx * 0.1
    return model


# ------------------------------------------------------------------ with_changes
def test_with_changes_returns_a_new_model_and_leaves_the_old_one_alone():
    model = _model()
    kx_before = model.kx.copy()
    new = model.with_changes(kx=model.kx * 2.0)
    assert np.allclose(new.kx, 2 * kx_before) and np.array_equal(model.kx, kx_before)
    assert new is not model


def test_with_changes_shares_nothing_with_the_old_model():
    model = _model()
    new = model.with_changes(porosity=0.25)
    new.kz[0] = -1.0
    new.pvt.bo[0] = 99.0
    new.producers[0].target = 1.0
    assert model.kz[0] > 0 and model.pvt.bo[0] != 99.0 and model.producers[0].target == 2500.0


def test_with_changes_accepts_one_number_or_one_value_per_block_like_the_constructor():
    model = _model()
    assert np.all(model.with_changes(porosity=0.3).porosity == 0.3)
    values = np.linspace(0.1, 0.3, model.grid.n_cells)
    assert np.allclose(model.with_changes(porosity=values).porosity, values)
    with pytest.raises(ValueError, match="porosity"):
        model.with_changes(porosity=[0.1, 0.2])


def test_with_changes_follows_paths_into_nested_objects_and_lists():
    model = _model()
    new = model.with_changes({"initial.woc_depth": 250.0, "producers[0].skin": 3.0, "injectors[0].target": 80.0})
    assert new.initial.woc_depth == 250.0 and new.producers[0].skin == 3.0 and new.injectors[0].target == 80.0
    assert model.initial.woc_depth == 200.0 and model.producers[0].skin == 0.0
    assert new.get("producers[0].skin") == 3.0


def test_with_changes_rejects_unknown_names_and_bad_paths():
    model = _model()
    with pytest.raises(AttributeError, match="permeability"):
        model.with_changes(permeability=1.0)
    with pytest.raises(AttributeError, match="skinn"):
        model.with_changes({"producers[0].skinn": 1.0})
    with pytest.raises(ValueError, match="cannot read"):
        model.with_changes({"producers[0]..skin": 1.0})


def test_with_changes_validates_the_new_model():
    model = _model()
    with pytest.raises(ValueError, match="dt"):
        model.with_changes(dt=-1.0)


def test_with_changes_records_provenance():
    model = _model()
    a = model.with_changes(kx=model.kx * 1.1, note="permeability +10 %")
    b = a.with_changes(porosity=0.25)
    assert model.provenance == []
    assert a.provenance == ["permeability +10 %"]
    assert b.provenance == ["permeability +10 %", "changed porosity"]


def test_an_edited_model_runs():
    from kusim import Simulator
    new = _model().with_changes(kx=200.0 * 3, ky=200.0 * 3, kz=100.0)
    res = Simulator(new).run()
    assert np.all(np.isfinite(res.pressure))


# ------------------------------------------------------------------ ParameterSet
def _perm_params():
    return ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=5000.0, ties=("ky", "kz"))])


def test_vector_and_apply_round_trip():
    model = _model()
    model = model.with_changes(kx=np.linspace(50.0, 900.0, model.grid.n_cells))
    params = ParameterSet([FieldParameter("kx", "log", ties=("ky", "kz")),
                           FieldParameter("porosity", "logit", lower=0.02, upper=0.4),
                           ScalarParameter("initial.woc_depth"), ScalarParameter("rock_compressibility", "log")])
    y = params.vector(model)
    assert y.shape == (params.size(model),) and params.size(model) == 2 * model.grid.n_cells + 2
    back = params.apply(model, y)
    assert np.allclose(back.kx, model.kx) and np.allclose(back.porosity, model.porosity)
    assert back.initial.woc_depth == model.initial.woc_depth
    assert np.isclose(back.rock_compressibility, model.rock_compressibility)


def test_the_vector_holds_log_permeability():
    model = _model()
    y = _perm_params().vector(model)
    assert np.allclose(y, np.log(200.0))


def test_an_additive_update_cannot_make_permeability_negative():
    model = _model()
    new = _perm_params().apply(model, np.full(model.grid.n_cells, -50.0))      # log k = -50
    assert np.all(new.kx > 0) and np.all(new.kx >= 1.0)                        # clipped at the lower bound


def test_values_are_clipped_to_the_physical_bounds():
    model = _model()
    params = _perm_params()
    low = params.apply(model, np.full(model.grid.n_cells, np.log(0.01)))
    high = params.apply(model, np.full(model.grid.n_cells, np.log(1e6)))
    assert np.allclose(low.kx, 1.0) and np.allclose(high.kx, 5000.0)


def test_tied_fields_keep_their_ratio():
    model = _model()
    y = _perm_params().vector(model)
    new = _perm_params().apply(model, y + np.log(3.0))
    assert np.allclose(new.kx, 600.0)
    assert np.allclose(new.ky, 0.8 * new.kx) and np.allclose(new.kz, 0.1 * new.kx)


def test_logit_keeps_porosity_inside_its_limits_for_any_update():
    model = _model()
    params = ParameterSet([FieldParameter("porosity", "logit", lower=0.05, upper=0.35)])
    for shift in (-40.0, -3.0, 0.0, 3.0, 40.0):
        new = params.apply(model, params.vector(model) + shift)
        assert np.all(new.porosity >= 0.05) and np.all(new.porosity <= 0.35) and np.all(np.isfinite(new.porosity))


def test_a_scalar_parameter_edits_the_nested_value():
    model = _model()
    params = ParameterSet([ScalarParameter("producers[0].skin"), ScalarParameter("initial.woc_depth", lower=100.0, upper=300.0)])
    new = params.apply(model, np.array([4.0, 1000.0]))
    assert new.producers[0].skin == 4.0 and new.initial.woc_depth == 300.0       # clipped
    assert model.producers[0].skin == 0.0


def test_bounds_are_given_in_the_transformed_space():
    model = _model()
    params = ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=5000.0),
                           FieldParameter("porosity", "logit", lower=0.05, upper=0.35),
                           ScalarParameter("initial.woc_depth", lower=100.0)])
    lo, hi = params.bounds(model)
    n = model.grid.n_cells
    assert np.allclose(lo[:n], 0.0) and np.allclose(hi[:n], np.log(5000.0))
    assert np.all(np.isinf(lo[n:2 * n])) and np.all(np.isinf(hi[n:2 * n]))        # logit: unbounded there
    assert lo[-1] == 100.0 and hi[-1] == np.inf


def test_ensemble_matrix_and_apply_all():
    model = _model()
    params = _perm_params()
    rng = np.random.default_rng(1)
    Y = params.vector(model)[:, None] + rng.normal(0, 0.3, size=(model.grid.n_cells, 6))
    models = params.apply_all(model, Y, note="member")
    assert len(models) == 6 and models[0].provenance == ["member"]
    assert np.allclose(params.matrix(models), Y)


def test_apply_all_can_edit_each_member_from_its_own_model():
    model = _model()
    params = _perm_params()
    members = [model.with_changes(kx=100.0, ky=80.0, kz=10.0), model.with_changes(kx=400.0, ky=160.0, kz=80.0)]
    Y = params.matrix(members) + np.log(2.0)
    new = params.apply_all(members, Y)
    assert np.allclose(new[0].kx, 200.0) and np.allclose(new[0].ky, 160.0)       # ratio 0.8 of member 0
    assert np.allclose(new[1].kx, 800.0) and np.allclose(new[1].ky, 320.0)       # ratio 0.4 of member 1
    with pytest.raises(ValueError, match="base models"):
        params.apply_all(members, np.zeros((model.grid.n_cells, 3)))


def test_apply_checks_the_vector():
    model = _model()
    params = _perm_params()
    with pytest.raises(ValueError, match="values"):
        params.apply(model, np.zeros(3))
    bad = params.vector(model)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        params.apply(model, bad)


def test_parameter_definitions_are_checked():
    with pytest.raises(ValueError, match="logit"):
        FieldParameter("porosity", "logit")                         # needs both bounds
    with pytest.raises(ValueError, match="transform"):
        FieldParameter("kx", "sqrt")
    with pytest.raises(ValueError, match="unique"):
        ParameterSet([FieldParameter("kx"), FieldParameter("kx")])
    with pytest.raises(ValueError, match="positive"):
        ParameterSet([FieldParameter("kx")]).vector(_model().with_changes(kx=0.0))
    with pytest.raises(ValueError, match="one value per block"):
        ParameterSet([FieldParameter("rock_compressibility")]).vector(_model())


# ------------------------------------------------------------------ pullback (gradients with respect to the vector)
def _fd_through_apply(params, model, y, weights, scalars=None, h=1e-6):
    """Central finite differences of J(y) = sum_f weights[f] . field_f(apply(y)) (+ scalar terms)."""
    def J(vec):
        new = params.apply(model, vec)
        total = sum(float(np.dot(w, new.get(f))) for f, w in weights.items())
        return total + sum(g * new.get(path) for path, g in (scalars or {}).items())
    grad = np.zeros_like(y)
    for i in range(y.size):
        yp, ym = y.copy(), y.copy()
        yp[i] += h
        ym[i] -= h
        grad[i] = (J(yp) - J(ym)) / (2 * h)
    return grad


def test_pullback_matches_finite_differences_through_apply():
    model = _model()
    n = model.grid.n_cells
    rng = np.random.default_rng(4)
    params = ParameterSet([FieldParameter("kx", "log", lower=1.0, upper=5000.0, ties=("ky", "kz")),
                           FieldParameter("porosity", "logit", lower=0.02, upper=0.4)])
    model = model.with_changes(kx=rng.uniform(50, 800, n), porosity=rng.uniform(0.1, 0.3, n))
    model = model.with_changes(ky=model.kx * 0.8, kz=model.kx * 0.1)
    y = params.vector(model)
    weights = {f: rng.normal(size=n) for f in ("kx", "ky", "kz", "porosity")}
    fd = _fd_through_apply(params, model, y, weights)
    analytic = params.pullback(model, y, weights)
    assert np.allclose(analytic, fd, rtol=1e-6, atol=1e-6)


def test_pullback_gives_zero_where_a_value_is_clipped_and_follows_the_identity_transform():
    model = _model()
    n = model.grid.n_cells
    params = ParameterSet([FieldParameter("porosity", "identity", lower=0.1, upper=0.3)])
    y = np.linspace(0.05, 0.35, n)                              # the two ends are outside the bounds
    weights = {"porosity": np.ones(n)}
    g = params.pullback(model, y, weights)
    inside = (y >= 0.1) & (y <= 0.3)
    assert np.all(g[inside] == 1.0) and np.all(g[~inside] == 0.0)
    assert np.allclose(g, _fd_through_apply(params, model, y, weights), atol=1e-6)


def test_pullback_of_a_scalar_needs_its_gradient():
    model = _model()
    params = ParameterSet([ScalarParameter("producers[0].skin", lower=-2.0, upper=10.0)])
    assert params.pullback(model, np.array([3.0]), {}, {"producers[0].skin": 2.5})[0] == 2.5
    assert params.pullback(model, np.array([30.0]), {}, {"producers[0].skin": 2.5})[0] == 0.0       # clipped
    with pytest.raises(NotImplementedError, match="finite differences"):
        params.pullback(model, np.array([3.0]), {})
    with pytest.raises(KeyError, match="kx"):
        ParameterSet([FieldParameter("kx")]).pullback(model, ParameterSet([FieldParameter("kx")]).vector(model), {})


# ------------------------------------------------------------------ datum depth of the equilibrium start
def test_datum_depth_moves_the_pressure_reference_not_the_state():
    from kusim import Grid, Simulator
    model = _model()
    g = model.grid
    # the same reservoir 500 ft deeper, described with absolute depths: datum and contact move by 500 ft too
    deeper = model.with_changes(grid=Grid(g.dx[:g.nx], g.dy[:g.n_layer:g.nx], g.dz[::g.n_layer], top_depth=500.0))
    deeper.initial = Equilibrium(datum_pressure=3000.0, woc_depth=700.0, datum_depth=500.0)
    a, b = Simulator(model), Simulator(deeper)
    assert np.allclose(a.p_o, b.p_o) and np.allclose(a.s_w, b.s_w)
    # a datum 100 ft deeper with the same pressure there means 0.433 * gamma_o * 100 psi less everywhere
    lower = model.with_changes({"initial.datum_depth": 100.0})
    assert np.allclose(Simulator(lower).p_o, a.p_o - 0.433 * model.gamma_o * 100.0)


def test_datum_depth_round_trips_and_is_optional_when_reading():
    from kusim import ReservoirModel
    model = _model().with_changes({"initial.datum_depth": 250.0})
    data = model.to_dict()
    assert data["initial"]["datum_depth"] == 250.0
    assert ReservoirModel.from_dict(data).initial.datum_depth == 250.0
    del data["initial"]["datum_depth"]
    assert ReservoirModel.from_dict(data).initial.datum_depth == 0.0
