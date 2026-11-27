"""Tests for Model.auxiliary_data: the global observables of the constraint terms.

Each constraint term of a likelihood measures its parameters through a global
observable (auxiliary datum): the centre of a HistFactory modifier's
constraint, or the variate of an auxiliary distribution such as the lumi
Gaussian of ``from_hifa``.  ``Model.auxiliary_data`` names them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.traversal import ancestors
from pytensor.tensor.variable import TensorConstant

import pyhs3

# SR and CR share the normsys bkg_norm; SR has a BB-lite staterror group, CR a
# Poisson shapesys with one unconstrained (zero-uncertainty) bin, and every
# sample carries lumi, whose Gaussian constraint is an auxiliary distribution.
SPEC: dict[str, Any] = {
    "channels": [
        {
            "name": "SR",
            "samples": [
                {
                    "name": "signal",
                    "data": [5.0, 10.0],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None},
                        {"name": "lumi", "type": "lumi", "data": None},
                    ],
                },
                {
                    "name": "background",
                    "data": [50.0, 60.0],
                    "modifiers": [
                        {
                            "name": "bkg_norm",
                            "type": "normsys",
                            "data": {"hi": 1.1, "lo": 0.9},
                        },
                        {"name": "stat_SR", "type": "staterror", "data": [5.0, 4.0]},
                        {"name": "lumi", "type": "lumi", "data": None},
                    ],
                },
            ],
        },
        {
            "name": "CR",
            "samples": [
                {
                    "name": "background",
                    "data": [100.0, 80.0, 60.0],
                    "modifiers": [
                        {
                            "name": "bkg_norm",
                            "type": "normsys",
                            "data": {"hi": 1.1, "lo": 0.9},
                        },
                        {
                            "name": "sys_CR",
                            "type": "shapesys",
                            "data": [10.0, 0.0, 5.0],
                        },
                        {"name": "lumi", "type": "lumi", "data": None},
                    ],
                },
            ],
        },
    ],
    "observations": [
        {"name": "SR", "data": [52.0, 75.0]},
        {"name": "CR", "data": [101.0, 77.0, 66.0]},
    ],
    "measurements": [
        {
            "name": "meas",
            "config": {
                "poi": "mu",
                "parameters": [
                    {
                        "name": "lumi",
                        "auxdata": [1.02],
                        "sigmas": [0.017],
                        "inits": [1.0],
                        "bounds": [[0.5, 1.5]],
                    }
                ],
            },
        }
    ],
    "version": "1.0.0",
}

# Observed auxiliary data of SPEC, keyed by constraint name: 0 for the
# normsys, 1 per bin for the staterror group, tau = (nominal / sigma)**2 for
# the constrained shapesys bins, the configured auxdata for lumi.
SPEC_AUXILIARY_DATA = {
    "constraint_bkg_norm": np.array(0.0),
    "constraint_bblite_model_SR": np.array([1.0, 1.0]),
    "constraint_sys_CR": np.array([100.0, 144.0]),
    "lumi_constraint_meas": np.array(1.02),
}


def spec_model() -> pyhs3.Model:
    ws = pyhs3.from_hifa(SPEC)
    return ws.model(ws.analyses["meas"], progress=False)


def hs3_workspace(
    modifiers: list[dict[str, Any]],
    *,
    aux: list[dict[str, Any]] | None = None,
    parameters: dict[str, float] | None = None,
    const: dict[str, float] | None = None,
    barlow_beeston_method: str = "lite",
    signal_modifiers: list[dict[str, Any]] | None = None,
) -> pyhs3.Workspace:
    """One two-bin HistFactory channel whose background carries ``modifiers``.

    The signal, scaled by ``mu``, also carries ``signal_modifiers``.  ``aux``
    distributions enter the likelihood as auxiliary distributions;
    ``parameters`` and ``const`` give the free and const parameter points
    beyond ``mu``, each free one with the domain [-5, 5] (``mu``: [0, 10]).
    """
    channel = {
        "name": "SR",
        "type": "histfactory_dist",
        "axes": [{"name": "obs_SR", "min": 0.0, "max": 1.0, "nbins": 2}],
        "barlow_beeston_method": barlow_beeston_method,
        "samples": [
            {
                "name": "signal",
                "data": {"contents": [5.0, 10.0], "errors": [1.0, 2.0]},
                "modifiers": [
                    {"name": "mu", "type": "normfactor", "parameter": "mu"},
                    *(signal_modifiers or []),
                ],
            },
            {
                "name": "background",
                "data": {"contents": [50.0, 80.0], "errors": [5.0, 4.0]},
                "modifiers": modifiers,
            },
        ],
    }
    points = [{"name": "mu", "value": 1.0}]
    points += [{"name": n, "value": v} for n, v in (parameters or {}).items()]
    points += [{"name": n, "value": v, "const": True} for n, v in (const or {}).items()]
    likelihood: dict[str, Any] = {
        "name": "L",
        "distributions": ["SR"],
        "data": ["obs_data_SR"],
    }
    if aux:
        likelihood["aux_distributions"] = [d["name"] for d in aux]
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[channel, *(aux or [])],
        data=[
            {
                "name": "obs_data_SR",
                "type": "binned",
                "contents": [57.0, 88.0],
                "axes": [{"name": "obs_SR", "min": 0.0, "max": 1.0, "nbins": 2}],
            }
        ],
        domains=[
            {
                "name": "main",
                "type": "product_domain",
                "axes": [
                    {"name": "mu", "min": 0.0, "max": 10.0},
                    *({"name": n, "min": -5.0, "max": 5.0} for n in (parameters or {})),
                ],
            }
        ],
        parameter_points=[{"name": "init", "parameters": points}],
        likelihoods=[likelihood],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
    )


def hs3_model(*args: Any, **kwargs: Any) -> pyhs3.Model:
    ws = hs3_workspace(*args, **kwargs)
    return ws.model(ws.analyses["A"], progress=False)


def normsys(constraint: str, name: str = "alpha") -> dict[str, Any]:
    return {
        "name": name,
        "type": "normsys",
        "parameter": name,
        "constraint": constraint,
        "data": {"hi": 1.1, "lo": 0.9},
    }


def shapesys(constraint: str) -> dict[str, Any]:
    return {
        "name": "sys",
        "type": "shapesys",
        "parameters": ["gamma_0", "gamma_1"],
        "constraint": constraint,
        "data": {"vals": [10.0, 4.0]},
    }


def staterror(constraint: str, *, full: bool) -> dict[str, Any]:
    modifier: dict[str, Any] = {
        "name": "stat",
        "type": "staterror",
        "parameters": ["gamma_0", "gamma_1"],
        "constraint": constraint,
    }
    if full:
        modifier["data"] = {"uncertainties": [5.0, 4.0]}
    return modifier


GAMMAS = {"gamma_0": 1.0, "gamma_1": 1.0}


class TestFromHifa:
    def test_names_shapes_and_values(self):
        auxiliary = spec_model().auxiliary_data

        assert set(auxiliary) == set(SPEC_AUXILIARY_DATA)
        for name, expected in SPEC_AUXILIARY_DATA.items():
            assert auxiliary[name].dtype == np.float64
            assert auxiliary[name].shape == expected.shape, name
            assert_allclose(auxiliary[name], expected, rtol=1e-12, err_msg=name)

    def test_returns_copies(self):
        model = spec_model()
        model.auxiliary_data["constraint_sys_CR"][0] = -1.0
        assert model.auxiliary_data["constraint_sys_CR"][0] == pytest.approx(100.0)

    def test_no_constraints(self):
        ws = pyhs3.from_hifa(
            {
                **SPEC,
                "channels": [
                    {
                        "name": "SR",
                        "samples": [
                            {
                                "name": "signal",
                                "data": [5.0, 10.0],
                                "modifiers": [
                                    {"name": "mu", "type": "normfactor", "data": None}
                                ],
                            }
                        ],
                    }
                ],
                "observations": [{"name": "SR", "data": [52.0, 75.0]}],
                "measurements": [
                    {"name": "meas", "config": {"poi": "mu", "parameters": []}}
                ],
            }
        )
        model = ws.model(ws.analyses["meas"], progress=False)
        assert model.auxiliary_data == {}

    def test_requires_a_likelihood(self):
        ws = hs3_workspace([normsys("Gauss")], parameters={"alpha": 0.0})
        with pytest.raises(RuntimeError, match="likelihood"):
            _ = ws.model(0, progress=False).auxiliary_data


class TestConstraintTypes:
    @pytest.mark.parametrize(
        ("constraint", "value"),
        [("Gauss", 0.0), ("Poisson", 1.0), ("LogNormal", 1.0)],
    )
    def test_single_parameter(self, constraint, value):
        model = hs3_model([normsys(constraint)], parameters={"alpha": 0.5})
        auxiliary = model.auxiliary_data
        assert list(auxiliary) == ["constraint_alpha"]
        assert auxiliary["constraint_alpha"].shape == ()
        assert auxiliary["constraint_alpha"] == value

    @pytest.mark.parametrize(
        ("constraint", "value"),
        # Relative uncertainties 0.2 and 0.05; tau = 1 / sigma_rel**2.
        [("Gauss", [1.0, 1.0]), ("Poisson", [25.0, 400.0]), ("LogNormal", [1.0, 1.0])],
    )
    def test_shapesys(self, constraint, value):
        model = hs3_model([shapesys(constraint)], parameters=GAMMAS)
        assert_allclose(model.auxiliary_data["constraint_sys"], value, rtol=1e-12)

    @pytest.mark.parametrize(
        ("constraint", "value"),
        [("Gauss", [1.0, 1.0]), ("Poisson", [100.0, 400.0]), ("LogNormal", [1.0, 1.0])],
    )
    def test_staterror_full(self, constraint, value):
        # BB-full uses this sample's own yields (50, 80) and errors (5, 4).
        model = hs3_model(
            [staterror(constraint, full=True)],
            parameters=GAMMAS,
            barlow_beeston_method="full",
        )
        assert_allclose(model.auxiliary_data["constraint_stat"], value, rtol=1e-12)

    @pytest.mark.parametrize(
        ("constraint", "value"),
        [("Gauss", [1.0, 1.0]), ("Poisson", [100.0, 400.0]), ("LogNormal", [1.0, 1.0])],
    )
    def test_staterror_lite(self, constraint, value):
        model = hs3_model([staterror(constraint, full=False)], parameters=GAMMAS)
        assert_allclose(model.auxiliary_data["constraint_bblite_SR"], value, rtol=1e-12)

    def test_duplicate_constraint_names_stay_distinct(self):
        """BB-full staterror modifiers of one name on two samples each keep their term."""
        model = hs3_model(
            [staterror("Gauss", full=True)],
            parameters=GAMMAS,
            barlow_beeston_method="full",
            signal_modifiers=[staterror("Gauss", full=True)],
        )
        auxiliary = model.auxiliary_data
        assert set(auxiliary) == {"constraint_stat", "constraint_stat_1"}
        assert_allclose(auxiliary["constraint_stat"], [1.0, 1.0])
        assert_allclose(auxiliary["constraint_stat_1"], [1.0, 1.0])


class TestAuxiliaryDistributions:
    def test_variate_is_a_literal(self):
        aux = {
            "name": "c",
            "type": "gaussian_dist",
            "x": 0.3,
            "mean": "alpha",
            "sigma": 1.0,
        }
        model = hs3_model([normsys("Gauss")], aux=[aux], parameters={"alpha": 0.0})
        assert model.auxiliary_data["c"] == pytest.approx(0.3)

    def test_variate_is_a_const_parameter(self):
        aux = {"name": "c", "type": "poisson_dist", "x": "nom_k", "mean": "alpha"}
        model = hs3_model(
            [normsys("Gauss")],
            aux=[aux],
            parameters={"alpha": 1.0},
            const={"nom_k": 3.0},
        )
        assert model.auxiliary_data["c"] == pytest.approx(3.0)

    def test_mean_is_the_global_observable(self):
        """ROOT writes Gauss(alpha | nom_alpha, 1): the const mean is the observable."""
        aux = {
            "name": "c",
            "type": "gaussian_dist",
            "x": "alpha",
            "mean": "nom_alpha",
            "sigma": 1.0,
        }
        model = hs3_model(
            [normsys("Gauss")],
            aux=[aux],
            parameters={"alpha": 0.0},
            const={"nom_alpha": 0.25},
        )
        assert model.auxiliary_data["c"] == pytest.approx(0.25)

    def test_no_global_observable(self):
        """A term whose variate is a free parameter has no auxiliary datum."""
        aux = {
            "name": "c",
            "type": "gaussian_dist",
            "x": "alpha",
            "mean": "beta",
            "sigma": 1.0,
        }
        model = hs3_model(
            [normsys("Gauss")], aux=[aux], parameters={"alpha": 0.0, "beta": 0.0}
        )
        assert "c" not in model.auxiliary_data

    def test_unsampleable_type(self):
        aux = {"name": "c", "type": "exponential_dist", "x": 1.0, "c": "alpha"}
        model = hs3_model([normsys("Gauss")], aux=[aux], parameters={"alpha": -1.0})
        assert set(model.auxiliary_data) == {"constraint_alpha"}


def test_global_observables_stay_distinct_in_log_prob():
    """Equal constants (the many 0s and 1s) are not merged into the observables."""
    model = spec_model()
    constants = {
        var.name
        for var in ancestors([model.log_prob])
        if isinstance(var, TensorConstant)
    }
    assert {
        "constraint_bkg_norm_x",
        "constraint_bblite_model_SR_x",
        "constraint_sys_CR_x",
    } <= constants


class TestSharedGlobalObservables:
    def test_observable_of_two_terms(self):
        """A const parameter that is the observable of two terms belongs to the first."""
        first = {
            "name": "c1",
            "type": "gaussian_dist",
            "x": "nom",
            "mean": "alpha",
            "sigma": 1.0,
        }
        second = {
            "name": "c2",
            "type": "gaussian_dist",
            "x": "nom",
            "mean": "beta",
            "sigma": 2.0,
        }
        model = hs3_model(
            [normsys("Gauss")],
            aux=[first, second],
            parameters={"alpha": 0.0, "beta": 0.0},
            const={"nom": 0.1},
        )
        assert set(model.auxiliary_data) == {"constraint_alpha", "c1"}


def product_workspace() -> pyhs3.Workspace:
    """An unbinned ProductDist channel: a Gaussian in x times a constraint on mu."""
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "shape",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": 1.0,
            },
            {
                "name": "mu_constraint",
                "type": "gaussian_dist",
                "x": "mu",
                "mean": "nom_mu",
                "sigma": 0.5,
            },
            {
                "name": "joint",
                "type": "product_dist",
                "factors": ["shape", "mu_constraint"],
            },
        ],
        data=[
            {
                "name": "events",
                "type": "unbinned",
                "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                "entries": [[0.5], [1.5]],
            }
        ],
        domains=[
            {
                "name": "main",
                "type": "product_domain",
                "axes": [{"name": "mu", "min": -5.0, "max": 5.0}],
            }
        ],
        parameter_points=[
            {
                "name": "init",
                "parameters": [
                    {"name": "mu", "value": 0.0},
                    {"name": "nom_mu", "value": 0.2, "const": True},
                ],
            }
        ],
        likelihoods=[{"name": "L", "distributions": ["joint"], "data": ["events"]}],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
    )


def test_product_constraint_factor():
    ws = product_workspace()
    model = ws.model(ws.analyses["A"], progress=False)
    assert model.auxiliary_data == {"mu_constraint": pytest.approx(0.2)}
