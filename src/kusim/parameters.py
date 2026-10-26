"""Parameters of a history match: which data of a model are uncertain, and how they map to a vector.

Ensemble methods (such as the Ensemble Kalman Filter) work on a **vector** of numbers per ensemble member.
A ``ParameterSet`` converts between a model and that vector, in both directions:

    params = ParameterSet([
        FieldParameter("kx", transform="log", lower=1.0, ties=("ky", "kz")),    # one number per block
        FieldParameter("porosity", transform="logit", lower=0.02, upper=0.40),
        ScalarParameter("initial.woc_depth"),                                    # one number
    ])
    y = params.vector(model)                  # the model as a vector (log-permeabilities, ...)
    new_model = params.apply(model, y + dy)   # a new model with the updated values; ``model`` is not changed

The vector holds the values in the **transformed** space: permeability enters as ``log(k)``, so that an additive
update (the Kalman update) can never make it negative.  ``apply`` transforms back and keeps every value inside the
physical bounds.  ``ties`` names other fields that keep their ratio to this one in the model that is edited
(``ky`` and ``kz`` as fixed multiples of ``kx``: change ``kx``, and they follow).

For an ensemble, ``matrix(models)`` stacks the vectors as columns and ``apply_all(base, matrix)`` makes one new
model per column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

TRANSFORMS = ("identity", "log", "logit")


def _forward(transform, x, lower, upper):
    if transform == "identity":
        return x
    if transform == "log":
        if np.any(x <= 0):
            raise ValueError("a log-transformed parameter must be positive")
        return np.log(x)
    # logit: maps the open interval (lower, upper) to the whole real line
    inside = np.clip(x, lower + 1e-12 * (upper - lower), upper - 1e-12 * (upper - lower))
    return np.log((inside - lower) / (upper - inside))


def _inverse(transform, y, lower, upper):
    if transform == "identity":
        return y
    if transform == "log":
        return np.exp(y)
    return lower + (upper - lower) / (1.0 + np.exp(-y))


def _inverse_slope(transform, y, values, lower, upper):
    """d(value)/d(y) of the inverse transform, at the transformed value ``y`` (``values`` = the physical value)."""
    if transform == "identity":
        return np.ones_like(values)
    if transform == "log":
        return values
    return (values - lower) * (upper - values) / (upper - lower)          # logit


def _transformed_bound(p, value, default):
    """A physical bound as a number in the transformed space (``default`` if there is none)."""
    if value is None or p.transform == "logit":
        return default
    return float(_forward(p.transform, np.array([value], dtype=float), p.lower, p.upper)[0])


def _check(transform, lower, upper, label):
    if transform not in TRANSFORMS:
        raise ValueError(f"{label}: transform must be one of {TRANSFORMS}, got {transform!r}")
    if transform == "logit" and (lower is None or upper is None):
        raise ValueError(f"{label}: a logit transform needs both lower and upper bounds")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"{label}: lower must be below upper")


@dataclass
class FieldParameter:
    """A model array with one value per block (``kx``, ``ky``, ``kz``, ``porosity``).

    ``lower`` and ``upper`` are physical bounds (psi, md, fraction, ...); ``ties`` lists other per-block fields
    that keep their ratio to this one (for permeability: ``("ky", "kz")``).
    """

    field: str
    transform: str = "log"
    lower: Optional[float] = None
    upper: Optional[float] = None
    ties: Sequence[str] = ()
    name: Optional[str] = None

    def __post_init__(self):
        self.name = self.name or f"{self.transform}_{self.field}"
        _check(self.transform, self.lower, self.upper, self.name)


@dataclass
class ScalarParameter:
    """One number of the model, given by its path: ``"rock_compressibility"``, ``"initial.woc_depth"``,
    ``"producers[0].skin"`` (see ``ReservoirModel.with_changes``)."""

    path: str
    transform: str = "identity"
    lower: Optional[float] = None
    upper: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.name = self.name or f"{self.transform}_{self.path}"
        _check(self.transform, self.lower, self.upper, self.name)


class ParameterSet:
    """A list of parameters and the conversion between a model and its parameter vector."""

    def __init__(self, parameters):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("a ParameterSet needs at least one parameter")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"parameter names must be unique, got {names}")

    # ------------------------------------------------------------------
    def _read(self, model, p):
        value = model.get(p.field if isinstance(p, FieldParameter) else p.path)
        if isinstance(p, FieldParameter):
            value = np.asarray(value, dtype=float)
            if value.shape != (model.grid.n_cells,):
                raise ValueError(f"{p.name}: {p.field!r} is not an array with one value per block")
            return value
        return np.atleast_1d(np.asarray(value, dtype=float))

    def layout(self, model):
        """``{name: slice}``: where each parameter sits in the vector of this model."""
        out, start = {}, 0
        for p in self.parameters:
            size = self._read(model, p).size
            out[p.name] = slice(start, start + size)
            start += size
        return out

    def size(self, model):
        """Length of the parameter vector."""
        return sum(s.stop - s.start for s in self.layout(model).values())

    def vector(self, model):
        """The model as a vector, in the transformed space."""
        return np.concatenate([_forward(p.transform, self._read(model, p), p.lower, p.upper) for p in self.parameters])

    def bounds(self, model):
        """``(lower, upper)`` vectors in the transformed space (infinite where a parameter has no bound).

        A logit-transformed parameter has no bound there: the transform keeps it inside its limits."""
        lower, upper = [], []
        for p in self.parameters:
            n = self._read(model, p).size
            lower.append(np.full(n, _transformed_bound(p, p.lower, -np.inf)))
            upper.append(np.full(n, _transformed_bound(p, p.upper, np.inf)))
        return np.concatenate(lower), np.concatenate(upper)

    # ------------------------------------------------------------------
    def apply(self, model, vector, note=None):
        """A new model with the values of ``vector`` (transformed space); ``model`` is not changed.

        Values are transformed back and clipped to ``lower``/``upper``.  Tied fields follow their ratio in ``model``.
        """
        vector = np.asarray(vector, dtype=float)
        layout = self.layout(model)
        if vector.shape != (self.size(model),):
            raise ValueError(f"the vector has {vector.size} values, the parameters need {self.size(model)}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("the parameter vector contains NaN or infinite values")
        changes = {}
        for p in self.parameters:
            values = _inverse(p.transform, vector[layout[p.name]], p.lower, p.upper)
            if p.lower is not None or p.upper is not None:
                values = np.clip(values, p.lower, p.upper)
            if isinstance(p, ScalarParameter):
                changes[p.path] = float(values[0])
                continue
            changes[p.field] = values
            old = np.asarray(model.get(p.field), dtype=float)
            for tied in p.ties:
                ratio = np.divide(np.asarray(model.get(tied), dtype=float), old, out=np.ones_like(old), where=old != 0)
                changes[tied] = values * ratio
        return model.with_changes(changes, note=note or "parameters updated")

    def pullback(self, model, vector, field_gradients, scalar_gradients=None):
        """Gradient of an objective with respect to the parameter ``vector``, from its gradients with respect to the model.

        ``field_gradients`` maps a field name (``"kx"``, ``"ky"``, ``"kz"``, ``"porosity"``) to dJ/d(field), one value per
        block, as an adjoint computation gives them; ``scalar_gradients`` maps a path (``"producers[0].skin"``) to dJ/d(value).
        This is the transpose of ``apply``: the chain rule through the transform (``ln k`` -> ``k``), through the bounds
        (a clipped value has zero gradient) and through the ties (a change of ``kx`` also changes ``ky`` and ``kz``).
        Parameters without a gradient raise ``KeyError`` (for a field) or ``NotImplementedError`` (for a scalar):
        use finite differences for those.
        """
        vector = np.asarray(vector, dtype=float)
        layout = self.layout(model)
        scalar_gradients = scalar_gradients or {}
        out = np.zeros_like(vector)
        for p in self.parameters:
            y = vector[layout[p.name]]
            raw = _inverse(p.transform, y, p.lower, p.upper)
            values = np.clip(raw, p.lower, p.upper) if (p.lower is not None or p.upper is not None) else raw
            inside = np.ones_like(values) if p.lower is None and p.upper is None else                 ((raw >= (-np.inf if p.lower is None else p.lower)) & (raw <= (np.inf if p.upper is None else p.upper))).astype(float)
            slope = _inverse_slope(p.transform, y, values, p.lower, p.upper) * inside
            if isinstance(p, ScalarParameter):
                if p.path not in scalar_gradients:
                    raise NotImplementedError(f"no gradient for {p.path!r}: use finite differences for this parameter")
                out[layout[p.name]] = scalar_gradients[p.path] * slope
                continue
            if p.field not in field_gradients:
                raise KeyError(f"no gradient for the field {p.field!r}")
            old = np.asarray(model.get(p.field), dtype=float)
            g = np.asarray(field_gradients[p.field], dtype=float).copy()
            for tied in p.ties:
                ratio = np.divide(np.asarray(model.get(tied), dtype=float), old, out=np.ones_like(old), where=old != 0)
                g += ratio * np.asarray(field_gradients.get(tied, 0.0), dtype=float)
            out[layout[p.name]] = g * slope
        return out

    # ------------------------------------------------------------------
    def matrix(self, models):
        """Parameter vectors of an ensemble of models as the columns of a matrix (size x ensemble size)."""
        return np.column_stack([self.vector(m) for m in models])

    def apply_all(self, model, matrix, note=None):
        """One new model per column of ``matrix``.

        ``model`` is the base all members are edited from, or a list with one base per column (the members of an
        ensemble that already differ, each edited from its own current model)."""
        matrix = np.asarray(matrix, dtype=float)
        bases = list(model) if isinstance(model, (list, tuple)) else [model] * matrix.shape[1]
        if len(bases) != matrix.shape[1]:
            raise ValueError(f"{len(bases)} base models for {matrix.shape[1]} columns")
        return [self.apply(base, matrix[:, j], note) for j, base in enumerate(bases)]
