"""Small, checked edits of a model, and a readable difference between two models.

An agent (or a script) that wants to change one permeability should not have to rewrite the whole model file.
``set_path`` changes one value of a model dictionary by a dotted path (``kx``, ``producers.0.target``,
``pvt.bo.2``, ``initial.sw``); ``apply_edits`` applies several and checks the result by building the model again, so
an edit that makes the model invalid is rejected with the same strict messages as reading a file.
``diff_models`` lists what differs between two models, again by path.

The paths are those of the JSON format (``docs/json-format.md``): dictionary keys and list positions counted from 0
(this is the position in the list, not a block number: block numbers inside the file still count from 1).
"""

from __future__ import annotations

import copy
import json

import numpy as np

from .spec import SpecError, model_from_dict, model_to_dict


def brief(value, limit=10):
    """A long list of numbers as ``{"n", "min", "max"}`` so that answers stay short; anything else unchanged."""
    if isinstance(value, list) and len(value) > limit and all(isinstance(v, (int, float)) for v in value):
        return {"n": len(value), "min": min(value), "max": max(value)}
    return value


def parse_value(text):
    """A command-line value: JSON if it is JSON (``150``, ``null``, ``[1, 2]``, ``"bhp"``), otherwise the text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _steps(path):
    if not isinstance(path, str) or not path.strip() or any(part == "" for part in path.split(".")):
        raise SpecError(f"{path!r} is not a path (write it like producers.0.target)", str(path), "bad_path")
    return path.split(".")


def get_path(data, path):
    """The value at a dotted path (raises SpecError ``bad_path`` when it does not exist)."""
    here = data
    for n, step in enumerate(_steps(path)):
        done = ".".join(_steps(path)[:n]) or "(the model)"
        if isinstance(here, dict):
            if step not in here:
                raise SpecError(f"{path}: there is no key {step!r} in {done}; keys are {sorted(here)}", path, "bad_path",
                                keys=sorted(here))
            here = here[step]
        elif isinstance(here, list):
            try:
                here = here[int(step)]
            except (ValueError, IndexError):
                raise SpecError(f"{path}: {step!r} is not a valid position in the list {done} "
                                f"(it has {len(here)} items, counted from 0)", path, "bad_path", length=len(here)) from None
        else:
            raise SpecError(f"{path}: {done} is a single value, so it has no {step!r}"
                            " (a uniform array is stored as one number: set the whole value instead)", path, "bad_path")
    return here


def set_path(data, path, value):
    """Set the value at a dotted path in a dictionary (in place); returns the old value.

    The key or position must exist, except that a missing key of a dictionary may be added when its parent exists
    (an optional well option, for instance). Whether the key is allowed is checked when the model is rebuilt.
    """
    steps = _steps(path)
    try:
        parent = get_path(data, ".".join(steps[:-1])) if len(steps) > 1 else data
    except SpecError as error:                           # report the path the caller gave, not its parent
        raise SpecError(str(error), path, error.code, **error.details) from None
    last = steps[-1]
    if isinstance(parent, dict):
        old = parent.get(last)
        parent[last] = value
        return old
    if isinstance(parent, list):
        old = get_path(data, path)                       # raises bad_path when the position does not exist
        parent[int(last)] = value
        return old
    raise SpecError(f"{path}: the parent is a single value, so it has no {last!r}", path, "bad_path")


def apply_edits(data, assignments, merge=None):
    """Apply edits to a copy of a model dictionary and check the result.

    ``assignments`` is a list of ``(path, value)``; ``merge`` is an optional dictionary merged first (nested
    dictionaries are merged key by key, everything else, lists included, is replaced). Returns
    ``(model, normalised_dict, changes)`` where ``changes`` lists ``{"path", "old", "new"}``. Raises SpecError or
    ValueError if the edited model is not valid.
    """
    work = copy.deepcopy(data)
    changes = []

    def merge_into(target, patch, prefix):
        for key, value in patch.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge_into(target[key], value, path)
            else:
                changes.append({"path": path, "old": brief(target.get(key)), "new": brief(value)})
                target[key] = value

    if merge:
        if not isinstance(merge, dict):
            raise SpecError("the patch must be a JSON object", "", "not_an_object")
        merge_into(work, merge, "")
    for path, value in assignments:
        changes.append({"path": path, "old": brief(set_path(work, path, value)), "new": brief(value)})
    model = model_from_dict(work)                        # strict: unknown keys and bad values are rejected here
    return model, model_to_dict(model), changes


# ---------------------------------------------------------------------- difference
_MAX_LISTED = 10


def _is_numbers(value):
    return isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)


def _diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            here = f"{path}.{key}" if path else key
            if key not in a:
                out.append({"path": here, "a": None, "b": b[key], "note": "only in b"})
            elif key not in b:
                out.append({"path": here, "a": a[key], "b": None, "note": "only in a"})
            else:
                _diff(a[key], b[key], here, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        if _is_numbers(a) and _is_numbers(b):
            x, y = np.asarray(a, float), np.asarray(b, float)
            where = np.flatnonzero(x != y)
            if len(where) > _MAX_LISTED:
                out.append({"path": path, "n_changed": int(len(where)), "of": len(a),
                            "max_abs_change": float(np.abs(x - y).max())})
            else:
                out.extend({"path": f"{path}.{i}", "a": a[i], "b": b[i]} for i in where)
        else:
            for i, (u, v) in enumerate(zip(a, b)):
                _diff(u, v, f"{path}.{i}", out)
    elif a != b:
        out.append({"path": path, "a": brief(a), "b": brief(b)})


def diff_dicts(a, b):
    """The differences between two JSON-like structures as a list of ``{"path", "a", "b"}`` records."""
    out = []
    _diff(a, b, "", out)
    return out


def diff_models(model_a, model_b):
    """What differs between two ``ReservoirModel`` objects, by JSON path (provenance excluded)."""
    da, db = model_to_dict(model_a), model_to_dict(model_b)
    da.pop("provenance", None), db.pop("provenance", None)
    return diff_dicts(da, db)
