"""Integration tests: ``pyhs3.to_hifa`` against pyhf.

Round trips over the specs of ``tests/test_hifa_pyhf.py``:

- pyhf -> HS3 -> pyhf gives back the input JSON, as :func:`normalized` compares
  them: dicts compare without regard to key order, and lists in order, except
  each measurement's config ``parameters``, which compare as a mapping by
  name after dropping what carries no information (see :func:`normalized`).
- HS3 -> pyhf -> HS3, for each workspace ``from_hifa`` produced, gives back
  the same HS3 workspace.

pyhf loads every converted workspace and builds its model with the
``modifier_settings`` from ``hifa_modifier_settings``.

The ROOT-exported fixtures in ``tests/test_histfactory`` go through
``to_hifa`` to pyhf and are compared against their companion ``*_hifa.json``
pyhf models by ``logpdf`` at seeded points.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_hifa_pyhf import SPECS

from pyhs3 import Workspace, from_hifa, hifa_modifier_settings, to_hifa

pyhf = pytest.importorskip("pyhf")

RTOL = 1e-10
SEEDS = [0, 1, 2]
FIXTURES = Path(__file__).parent / "test_histfactory"

# pyhf's defaults for config ``inits`` and ``bounds`` per modifier type, for
# one scalar parameter or one bin (pyhf's ``schemas/1.0.0`` and
# ``pyhf.modifiers``); lumi has none.
PYHF_DEFAULTS: dict[str, tuple[float, tuple[float, float]]] = {
    "normfactor": (1.0, (0.0, 10.0)),
    "normsys": (0.0, (-5.0, 5.0)),
    "histosys": (0.0, (-5.0, 5.0)),
    "shapefactor": (1.0, (0.0, 10.0)),
    "shapesys": (1.0, (1e-10, 10.0)),
    "staterror": (1.0, (1e-10, 10.0)),
}


def modifier_types(spec: dict[str, Any]) -> dict[str, str]:
    """pyhf modifier name -> type, over every sample of every channel."""
    return {
        modifier["name"]: modifier["type"]
        for channel in spec["channels"]
        for sample in channel["samples"]
        for modifier in sample["modifiers"]
    }


def normalized(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with each measurement's config parameters as a mapping by name.

    An entry drops ``inits`` and ``bounds`` equal to pyhf's default for every
    bin, ``fixed: false``, and, except for ``lumi``, ``auxdata``, ``sigmas``
    and ``factors`` (``from_hifa`` accepts only the values pyhf derives, so
    they hold no information). Entries naming no parameter of the model, and
    entries left with only a name, are dropped.
    """
    types = modifier_types(spec)
    result = copy.deepcopy(spec)
    for measurement in result["measurements"]:
        entries: dict[str, dict[str, Any]] = {}
        for entry in measurement["config"]["parameters"]:
            name = entry["name"]
            if name not in types:
                continue
            kept = {k: v for k, v in entry.items() if k != "name"}
            if types[name] != "lumi":
                init, bounds = PYHF_DEFAULTS[types[name]]
                for key in ("auxdata", "sigmas", "factors"):
                    kept.pop(key, None)
                if all(v == init for v in kept.get("inits", [])):
                    kept.pop("inits", None)
                if all(tuple(b) == bounds for b in kept.get("bounds", [])):
                    kept.pop("bounds", None)
            if kept.get("fixed") is False:
                del kept["fixed"]
            if kept:
                entries[name] = kept
        measurement["config"]["parameters"] = entries
    return result


@pytest.mark.parametrize("name", list(SPECS))
def test_pyhf_to_hs3_to_pyhf(name):
    spec = SPECS[name]
    ws = from_hifa(spec)
    result = to_hifa(ws)
    assert normalized(result) == normalized(spec)
    # pyhf accepts the output and builds its model.
    pyhf.Workspace(result).model(modifier_settings=hifa_modifier_settings(ws))


@pytest.mark.parametrize("name", list(SPECS))
def test_hs3_to_pyhf_to_hs3(name):
    ws = from_hifa(SPECS[name])
    back = from_hifa(to_hifa(ws), modifier_settings=hifa_modifier_settings(ws))
    assert back.model_dump() == ws.model_dump()


@pytest.mark.parametrize(
    "settings",
    [
        {"normsys": {"interpcode": "code1"}, "histosys": {"interpcode": "code0"}},
        {"normsys": {"interpcode": "code4"}, "histosys": {"interpcode": "code2"}},
    ],
)
def test_interpolation_codes_round_trip(settings):
    ws = from_hifa(SPECS["shared_nuispar_across_types"], modifier_settings=settings)
    assert hifa_modifier_settings(ws) == settings
    back = from_hifa(to_hifa(ws), modifier_settings=hifa_modifier_settings(ws))
    assert back.model_dump() == ws.model_dump()


# The ROOT-exported HS3 fixtures, with the likelihood to convert. ROOT names a
# normsys/histosys parameter ``alpha_<modifier>`` and adds a normfactor
# ``Lumi``, const at 1 in its parameter points; pyhf names the parameter after
# the modifier and has no ``Lumi``. The analyses set no ``init``, so
# ``Lumi`` converts to a free normfactor, set to 1 at each point compared.
ROOT_FIXTURES = [
    pytest.param("simplemodel_correlated-background", id="correlated"),
    pytest.param("simplemodel_uncorrelated-background", id="uncorrelated"),
]


def root_to_pyhf_name(name: str) -> str:
    return name.removeprefix("alpha_")


@pytest.mark.parametrize("fixture", ROOT_FIXTURES)
@pytest.mark.parametrize("seed", SEEDS)
def test_root_fixture_logpdf(fixture, seed):
    ws = Workspace(**json.loads((FIXTURES / f"{fixture}_hs3.json").read_text()))
    converted = pyhf.Workspace(to_hifa(ws, likelihood="simPdf_obsData"))
    model = converted.model(
        modifier_settings=hifa_modifier_settings(ws, likelihood="simPdf_obsData")
    )
    data = converted.data(model)

    companion = pyhf.Workspace(
        json.loads((FIXTURES / f"{fixture}_hifa.json").read_text())
    )
    expected_model = companion.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        }
    )
    expected_data = companion.data(expected_model)
    assert list(data) == list(expected_data)

    rng = np.random.default_rng(seed)
    names = expected_model.config.par_names
    bounds = expected_model.config.suggested_bounds()
    for _ in range(4):
        # One value per companion parameter, inside the middle of its bounds.
        point = {
            name: lo + (hi - lo) * rng.uniform(0.3, 0.7)
            for name, (lo, hi) in zip(names, bounds, strict=True)
        }
        pars = [
            1.0 if name == "Lumi" else point[root_to_pyhf_name(name)]
            for name in model.config.par_names
        ]
        expected = expected_model.logpdf([point[name] for name in names], expected_data)
        assert model.logpdf(pars, data)[0] == pytest.approx(expected[0], rel=RTOL)
