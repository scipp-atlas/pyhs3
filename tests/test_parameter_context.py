"""
Tests for ParameterContext and Workspace.parameter_context.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyhs3 import Workspace
from pyhs3.domains import Domain, ProductDomain
from pyhs3.exceptions import UnlistedParameterError
from pyhs3.parameter_context import ParameterContext
from pyhs3.parameter_points import ParameterPoint, ParameterSet

_WS_DICT: dict = {
    "metadata": {"hs3_version": "0.2"},
    "distributions": [
        {
            "name": "sig",
            "type": "gaussian_dist",
            "x": "x",
            "mean": "mu",
            "sigma": "sigma",
        }
    ],
    "parameter_points": [
        {
            "name": "nominal",
            "parameters": [
                {"name": "mu", "value": 0.0},
                {"name": "sigma", "value": 1.0, "const": True},
            ],
        },
        {
            "name": "alt",
            "parameters": [
                {"name": "mu", "value": 1.0},
                {"name": "sigma", "value": 2.0},
            ],
        },
    ],
    "domains": [
        {
            "name": "first_domain",
            "type": "product_domain",
            "axes": [
                {"name": "mu", "min": -10.0, "max": 10.0},
                {"name": "sigma", "min": 0.1, "max": 10.0},
            ],
        },
        {
            "name": "second_domain",
            "type": "product_domain",
            "axes": [{"name": "nu", "min": 0.0}],
        },
    ],
    "data": [
        {
            "name": "obs",
            "type": "unbinned",
            "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
            "entries": [[0.5], [1.0]],
        }
    ],
    "likelihoods": [{"name": "L", "distributions": ["sig"], "data": ["obs"]}],
    "analyses": [
        {
            "name": "A",
            "likelihood": "L",
            "parameters_of_interest": ["mu"],
            "domains": ["first_domain"],
            "init": "nominal",
        },
        {
            "name": "merged",
            "likelihood": "L",
            "domains": ["first_domain", "second_domain"],
            "init": "nominal",
        },
    ],
}


@pytest.fixture
def ws() -> Workspace:
    return Workspace(**_WS_DICT)


class TestParameterContext:
    """The resolved views a ParameterContext gives of its parameter set and domain."""

    @pytest.fixture
    def context(self) -> ParameterContext:
        return ParameterContext(
            parameter_set=ParameterSet(
                name="point",
                parameters=[
                    ParameterPoint(name="mu", value=1.5),
                    ParameterPoint(name="lumi", value=1.0, const=True),
                ],
            ),
            domain=ProductDomain(
                name="dom",
                axes=[
                    {"name": "mu", "min": 0.0, "max": 5.0},
                    {"name": "lumi", "min": 0.9},
                    {"name": "fixed", "const": True},
                ],
            ),
            parameters_of_interest=("mu",),
        )

    def test_name_is_parameter_set_name(self, context):
        assert context.name == "point"

    def test_values(self, context):
        assert context.values == {"mu": 1.5, "lumi": 1.0}

    def test_const(self, context):
        assert context.const == frozenset({"lumi"})

    def test_bounds_skip_constant_axes(self, context):
        assert context.bounds == {"mu": (0.0, 5.0), "lumi": (0.9, np.inf)}

    def test_parameters_of_interest(self, context):
        assert context.parameters_of_interest == ("mu",)

    def test_frozen(self, context):
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.parameters_of_interest = ()  # type: ignore[misc]

    def test_defaults(self):
        context = ParameterContext(parameter_set=ParameterSet(name="p"))
        assert context.bounds == {}
        assert context.parameters_of_interest == ()
        assert context.values == {}
        assert context.const == frozenset()

    def test_bounds_of_untyped_domain_are_empty(self):
        context = ParameterContext(
            parameter_set=ParameterSet(name="p"),
            domain=Domain(name="d", type="unknown"),
        )
        assert context.bounds == {}


class TestWorkspaceParameterContextAnalysis:
    """Workspace.parameter_context for an analysis target."""

    def test_init_resolution(self, ws):
        context = ws.parameter_context(ws.analyses["A"])
        assert context.name == "nominal"
        assert context.values == {"mu": 0.0, "sigma": 1.0}
        assert context.const == frozenset({"sigma"})
        assert context.bounds == {"mu": (-10.0, 10.0), "sigma": (0.1, 10.0)}

    def test_parameters_of_interest(self, ws):
        assert ws.parameter_context(ws.analyses["A"]).parameters_of_interest == ("mu",)

    def test_no_parameters_of_interest(self, ws):
        assert ws.parameter_context(ws.analyses["merged"]).parameters_of_interest == ()

    def test_merged_domains(self, ws):
        context = ws.parameter_context(ws.analyses["merged"])
        assert context.bounds == {
            "mu": (-10.0, 10.0),
            "sigma": (0.1, 10.0),
            "nu": (0.0, np.inf),
        }

    @pytest.mark.parametrize("override", ["alt", 1])
    def test_parameter_set_override_replaces_init(self, ws, override):
        context = ws.parameter_context(ws.analyses["A"], parameter_set=override)
        assert context.name == "alt"
        assert context.values == {"mu": 1.0, "sigma": 2.0}
        assert context.const == frozenset()

    def test_parameter_set_instance_override(self, ws):
        custom = ParameterSet(
            name="custom", parameters=[ParameterPoint(name="mu", value=7.0)]
        )
        context = ws.parameter_context(ws.analyses["A"], parameter_set=custom)
        assert context.parameter_set is custom
        assert context.values == {"mu": 7.0}

    def test_no_init_uses_empty_parameter_set(self):
        ws = Workspace(
            **{
                **_WS_DICT,
                "analyses": [
                    {"name": "A", "likelihood": "L", "domains": ["first_domain"]}
                ],
            }
        )
        context = ws.parameter_context(ws.analyses["A"])
        assert context.name == "default"
        assert context.values == {}

    def test_domain_override_raises(self, ws):
        with pytest.raises(ValueError, match="domain override"):
            ws.parameter_context(ws.analyses["A"], domain="second_domain")

    def test_unknown_init_raises(self):
        ws = Workspace(
            **{
                **_WS_DICT,
                "analyses": [
                    {
                        "name": "A",
                        "likelihood": "L",
                        "domains": ["first_domain"],
                        "init": "missing",
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="unknown parameter set 'missing'"):
            ws.parameter_context(ws.analyses["A"])

    def test_section_2_7_violation_raises(self, ws):
        """sigma is outside the domain and only const in 'nominal', not in 'alt'."""
        ws = Workspace(
            **{
                **_WS_DICT,
                "analyses": [
                    {
                        "name": "A",
                        "likelihood": "L",
                        "domains": ["second_domain"],
                        "init": "nominal",
                    }
                ],
            }
        )
        with pytest.raises(UnlistedParameterError, match=r"section 2\.7") as excinfo:
            ws.parameter_context(ws.analyses["A"])
        assert excinfo.value.parameters == ["mu"]

    def test_section_2_7_checks_override(self):
        """The override replaces init as the referenced parameter point."""
        ws = Workspace(
            **{
                **_WS_DICT,
                "analyses": [
                    {
                        "name": "A",
                        "likelihood": "L",
                        "domains": ["second_domain"],
                    }
                ],
            }
        )
        fixed = ParameterSet(
            name="fixed",
            parameters=[
                ParameterPoint(name="mu", value=0.0, const=True),
                ParameterPoint(name="sigma", value=1.0, const=True),
            ],
        )
        with pytest.raises(UnlistedParameterError):
            ws.parameter_context(ws.analyses["A"])
        assert ws.parameter_context(ws.analyses["A"], parameter_set=fixed).const == {
            "mu",
            "sigma",
        }


class TestWorkspaceParameterContextOtherTargets:
    """Workspace.parameter_context for likelihood, name and domain-index targets."""

    def test_likelihood_defaults(self, ws):
        context = ws.parameter_context(ws.likelihoods["L"])
        assert context.name == "nominal"
        assert context.bounds == {"mu": (-10.0, 10.0), "sigma": (0.1, 10.0)}
        assert context.parameters_of_interest == ()

    def test_likelihood_overrides(self, ws):
        context = ws.parameter_context(
            ws.likelihoods["L"], parameter_set="alt", domain="second_domain"
        )
        assert context.name == "alt"
        assert context.bounds == {"nu": (0.0, np.inf)}

    def test_name_resolves_to_analysis(self, ws):
        assert ws.parameter_context("A").parameters_of_interest == ("mu",)

    def test_name_resolves_to_likelihood(self, ws):
        assert ws.parameter_context("L", domain=1).bounds == {"nu": (0.0, np.inf)}

    def test_name_falls_back_to_domain(self, ws):
        assert ws.parameter_context("second_domain").bounds == {"nu": (0.0, np.inf)}

    def test_name_analysis_domain_override_raises(self, ws):
        with pytest.raises(ValueError, match="domain override"):
            ws.parameter_context("A", domain=0)

    def test_domain_index(self, ws):
        context = ws.parameter_context(1, parameter_set="alt")
        assert context.name == "alt"
        assert context.bounds == {"nu": (0.0, np.inf)}

    def test_model_uses_context(self, ws):
        """Workspace.model builds from the same resolution as parameter_context."""
        model = ws.model(ws.analyses["A"], progress=False)
        assert model.free_params == {"mu": 0.0}
