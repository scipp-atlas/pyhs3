"""Checks an analysis must pass when its model is built, not when the workspace loads.

A workspace can hold several analyses with different configurations, so these
are checked only for the analysis a model is built from:

- every parameter of interest is a parameter of the analysis's likelihood;
- a per-bin HistFactory parameter whose bin gets no constraint term (zero
  uncertainty on a positive yield) is ``const``, as pyhf and ROOT fix it.
"""

from __future__ import annotations

from typing import Any

import pytest

from pyhs3 import Workspace
from pyhs3.distributions.histfactory.data import SampleData
from pyhs3.distributions.histfactory.modifiers import StatErrorModifier
from pyhs3.exceptions import (
    UnconstrainedParameterError,
    UnknownParameterOfInterestError,
)


def _histfactory_workspace(
    modifier: dict[str, Any],
    *,
    errors: list[float],
    points: list[dict[str, Any]],
    parameters_of_interest: list[str] | None = None,
    barlow_beeston_method: str = "lite",
) -> Workspace:
    """One two-bin channel with a ``mu`` normfactor and ``modifier`` on its sample.

    Bin 0 has a positive yield and a positive uncertainty, bin 1 a positive
    yield; ``modifier`` and ``errors`` decide bin 1's uncertainty.
    """
    gammas = modifier.get("parameters", [])
    axis = {"name": "obs_x_c", "min": 0.0, "max": 2.0, "nbins": 2}
    return Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "type": "histfactory_dist",
                "name": "model_c",
                "axes": [axis],
                "barlow_beeston_method": barlow_beeston_method,
                "samples": [
                    {
                        "name": "s",
                        "data": {"contents": [10.0, 20.0], "errors": errors},
                        "modifiers": [
                            {
                                "type": "normfactor",
                                "name": "mu",
                                "parameter": "mu",
                            },
                            modifier,
                        ],
                    }
                ],
            }
        ],
        data=[
            {
                "type": "binned",
                "name": "obsData_c",
                "axes": [axis],
                "contents": [11.0, 19.0],
            }
        ],
        likelihoods=[
            {"name": "L", "distributions": ["model_c"], "data": ["obsData_c"]}
        ],
        domains=[
            {
                "name": "D",
                "type": "product_domain",
                "axes": [
                    {"name": name, "min": 0.0, "max": 10.0} for name in ["mu", *gammas]
                ],
            }
        ],
        parameter_points=[{"name": "init", "parameters": points}],
        analyses=[
            {
                "name": "A",
                "likelihood": "L",
                "domains": ["D"],
                "init": "init",
                "parameters_of_interest": parameters_of_interest or ["mu"],
            }
        ],
    )


def _points(*, gamma_1_const: bool) -> list[dict[str, Any]]:
    return [
        {"name": "mu", "value": 1.0},
        {"name": "gamma_0", "value": 1.0},
        {"name": "gamma_1", "value": 1.0, "const": gamma_1_const},
    ]


SHAPESYS_ZERO_UNCERTAINTY = {
    "type": "shapesys",
    "name": "shapesys",
    "parameters": ["gamma_0", "gamma_1"],
    "data": {"vals": [2.0, 0.0]},
}
STATERROR = {
    "type": "staterror",
    "name": "staterror",
    "parameters": ["gamma_0", "gamma_1"],
}
STATERROR_FULL_ZERO_UNCERTAINTY = {
    **STATERROR,
    "data": {"uncertainties": [2.0, 0.0]},
}

#: (modifier, sample errors, Barlow-Beeston method) whose bin 1 has a
#: positive yield and zero uncertainty, so gamma_1 gets no constraint term.
UNCONSTRAINED_GAMMA_CASES = {
    "shapesys": (SHAPESYS_ZERO_UNCERTAINTY, [0.0, 0.0], "lite"),
    "staterror_lite": (STATERROR, [2.0, 0.0], "lite"),
    "staterror_full": (STATERROR_FULL_ZERO_UNCERTAINTY, [2.0, 0.0], "full"),
}


class TestParametersOfInterest:
    def test_unknown_poi_loads_but_raises_at_model_build(self):
        """A POI the likelihood does not use only fails for the analysis built."""
        ws = _histfactory_workspace(
            SHAPESYS_ZERO_UNCERTAINTY,
            errors=[0.0, 0.0],
            points=_points(gamma_1_const=True),
            parameters_of_interest=["mu_SIG"],
        )
        with pytest.raises(UnknownParameterOfInterestError, match="mu_SIG") as excinfo:
            ws.model(ws.analyses["A"], progress=False)
        assert excinfo.value.parameters == ["mu_SIG"]

    def test_const_poi_is_accepted(self):
        """A POI held const in the parameter point is still a parameter of the model."""
        points = [
            {**p, "const": True} if p["name"] == "mu" else p
            for p in _points(gamma_1_const=True)
        ]
        ws = _histfactory_workspace(
            SHAPESYS_ZERO_UNCERTAINTY, errors=[0.0, 0.0], points=points
        )
        model = ws.model(ws.analyses["A"], progress=False)
        assert model.context.parameters_of_interest == ("mu",)


class TestUnconstrainedGammaMustBeConst:
    @pytest.mark.parametrize("case", list(UNCONSTRAINED_GAMMA_CASES))
    def test_free_unconstrained_gamma_raises(self, case):
        modifier, errors, method = UNCONSTRAINED_GAMMA_CASES[case]
        ws = _histfactory_workspace(
            modifier,
            errors=errors,
            points=_points(gamma_1_const=False),
            barlow_beeston_method=method,
        )
        with pytest.raises(UnconstrainedParameterError, match="gamma_1") as excinfo:
            ws.model(ws.analyses["A"], progress=False)
        assert excinfo.value.parameters == ["gamma_1"]

    @pytest.mark.parametrize("case", list(UNCONSTRAINED_GAMMA_CASES))
    def test_const_unconstrained_gamma_builds(self, case):
        modifier, errors, method = UNCONSTRAINED_GAMMA_CASES[case]
        ws = _histfactory_workspace(
            modifier,
            errors=errors,
            points=_points(gamma_1_const=True),
            barlow_beeston_method=method,
        )
        model = ws.model(ws.analyses["A"], progress=False)
        assert "gamma_1" not in model.free_params

    def test_zero_yield_bin_gamma_may_float(self):
        """A zero-yield bin's gamma scales nothing, so it need not be const."""
        modifier = {**SHAPESYS_ZERO_UNCERTAINTY, "data": {"vals": [2.0, 3.0]}}
        ws = _histfactory_workspace(
            modifier, errors=[0.0, 0.0], points=_points(gamma_1_const=False)
        )
        zero_yield = ws.distributions["model_c"].samples[0].data
        zero_yield.contents[1] = 0.0
        model = ws.model(ws.analyses["A"], progress=False)
        assert "gamma_1" in model.free_params


def test_lite_staterror_modifier_reports_no_unconstrained_parameters():
    """Without per-modifier uncertainties (BB-lite), the channel decides, not the modifier."""
    modifier = StatErrorModifier(name="staterror", parameters=["gamma_0", "gamma_1"])
    sample = SampleData(contents=[10.0, 20.0], errors=[2.0, 0.0])
    assert modifier.unconstrained_parameters(sample) == []
