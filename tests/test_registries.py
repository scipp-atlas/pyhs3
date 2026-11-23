"""Registering a custom function or distribution type makes it loadable from JSON."""

from __future__ import annotations

import json
import math
from typing import Literal

import numpy as np
import pytensor.tensor as pt
import pytest
from pydantic import ValidationError

import pyhs3
from pyhs3.distributions import (
    Distribution,
    Distributions,
    GaussianDist,
    registered_distributions,
)
from pyhs3.functions import Function, Functions, registered_functions


class ScaledFunction(Function):
    """``x * factor``: a custom function type for these tests."""

    type: Literal["scaled_function"] = "scaled_function"
    x: str
    factor: float

    def _expression(self, context):
        return context[self.x] * self.factor


class FlatDist(Distribution):
    """A uniform density on the observable's domain: a custom distribution type."""

    type: Literal["flat_dist"] = "flat_dist"
    x: str

    def likelihood(self, context):
        return pt.ones_like(context[self.x])


@pytest.fixture
def registered(monkeypatch):
    """Register the custom types for one test; the registries are restored after."""
    monkeypatch.setitem(registered_functions, "scaled_function", ScaledFunction)
    monkeypatch.setitem(registered_distributions, "flat_dist", FlatDist)


def _workspace_dict() -> dict:
    return {
        "metadata": {"hs3_version": "0.2"},
        "functions": [
            {"name": "mean", "type": "scaled_function", "x": "mu", "factor": 2.0}
        ],
        "distributions": [
            {
                "name": "g",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mean",
                "sigma": 1.0,
            },
            {"name": "flat", "type": "flat_dist", "x": "x"},
        ],
        "parameter_points": [
            {
                "name": "p",
                "parameters": [
                    {"name": "mu", "value": 0.5},
                    {"name": "x", "value": 0.0},
                ],
            }
        ],
        "domains": [
            {
                "name": "d",
                "type": "product_domain",
                "axes": [
                    {"name": "x", "min": -5.0, "max": 5.0},
                    {"name": "mu", "min": -2.0, "max": 2.0},
                ],
            }
        ],
    }


@pytest.mark.usefixtures("registered")
class TestRegisteredTypes:
    def test_load_from_a_workspace_dict(self):
        ws = pyhs3.Workspace(**_workspace_dict())
        assert isinstance(ws.functions["mean"], ScaledFunction)
        assert isinstance(ws.distributions["flat"], FlatDist)

    def test_model_evaluates_registered_types(self):
        ws = pyhs3.Workspace(**_workspace_dict())
        model = ws.model(0, progress=False)

        density = model.pdf_unsafe("g", x=1.0, mu=0.5)
        flat = model.pdf_unsafe("flat", x=1.0, mu=0.5)

        # Gaussian at x = 1 with mean 2 * mu = 1. Without data, x is not an
        # observable, so neither density is renormalized over its domain.
        assert float(density) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
        np.testing.assert_allclose(flat, 1.0)

    def test_dump_keeps_the_registered_types_fields(self):
        dumped = json.loads(pyhs3.Workspace(**_workspace_dict()).model_dump_json())
        function = next(f for f in dumped["functions"] if f["name"] == "mean")
        assert function["type"] == "scaled_function"
        assert function["factor"] == 2.0

    def test_unknown_type_lists_the_registered_types(self):
        with pytest.raises(
            ValidationError, match="Unknown function type 'nope'"
        ) as info:
            Functions([{"name": "f", "type": "nope", "x": "a"}])
        assert "scaled_function" in str(info.value)

    def test_errors_inside_a_registered_type_name_the_field(self):
        with pytest.raises(ValidationError) as info:
            Functions(
                [{"name": "f", "type": "scaled_function", "x": "a", "factor": "big"}]
            )
        assert info.value.errors()[0]["loc"][-1] == "factor"


class TestInstances:
    def test_an_instance_of_a_subclass_passes_through(self):
        """Instances are kept as they are, without consulting the registry."""

        class Narrow(GaussianDist):
            pass

        dist = Narrow(name="g", x="x", mean="m", sigma=0.5)
        assert Distributions([dist])["g"] is dist

    def test_unregistered_types_are_rejected(self):
        with pytest.raises(
            ValidationError, match="Unknown distribution type 'flat_dist'"
        ):
            Distributions([{"name": "flat", "type": "flat_dist", "x": "x"}])
