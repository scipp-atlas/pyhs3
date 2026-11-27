"""Tests for Model.auxiliary_data: the global observables of the constraint terms.

Each constraint term of a likelihood measures its parameters through a global
observable (auxiliary datum): the centre of a HistFactory modifier's
constraint, or the variate of an auxiliary distribution such as the lumi
Gaussian of ``from_hifa``.  ``Model.auxiliary_data`` names them.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.traversal import ancestors
from pytensor.tensor.variable import TensorConstant
from scipy.special import gammaln
from scipy.stats import norm

import pyhs3
from pyhs3 import model as model_module

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

    def test_no_constrained_bin(self):
        """A shapesys whose every bin has zero uncertainty has no term and no datum."""
        modifier = {**shapesys("Poisson"), "data": {"vals": [0.0, 0.0]}}
        model = hs3_model(
            [normsys("Gauss"), modifier],
            parameters={"alpha": 0.0},
            const=GAMMAS,
        )
        assert set(model.auxiliary_data) == {"constraint_alpha"}

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


# Swapped auxiliary data of SPEC: every global observable off its observed value.
SWAPPED_AUXILIARY_DATA = {
    "constraint_bkg_norm": np.array(0.4),
    "constraint_bblite_model_SR": np.array([1.1, 0.95]),
    "constraint_sys_CR": np.array([93.0, 151.0]),
    "lumi_constraint_meas": np.array(0.99),
}

# Swapped observed counts of SPEC.
SWAPPED_COUNTS = {
    "model_SR": np.array([61.5, 70.0]),
    "model_CR": np.array([97.0, 83.0, 58.0]),
}


def spec_model_with_lumi_auxdata(auxdata: float) -> pyhs3.Model:
    """SPEC's model, with lumi's configured auxiliary datum replaced."""
    spec = copy.deepcopy(SPEC)
    spec["measurements"][0]["config"]["parameters"][0]["auxdata"] = [auxdata]
    ws = pyhs3.from_hifa(spec)
    return ws.model(ws.analyses["meas"], progress=False)


def log_poisson(k: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Continuous Poisson log-pmf, as the constraint terms evaluate it."""
    return k * np.log(rate) - rate - gammaln(k + 1.0)


def constraint_log_delta(point: dict[str, float]) -> float:
    """log L(swapped auxiliary data) - log L(observed auxiliary data) of SPEC at ``point``.

    Only the constraint terms depend on the auxiliary data; the per-bin
    rates use the observed tau (pyhf's convention), not the swapped datum.
    """
    swapped, observed = SWAPPED_AUXILIARY_DATA, SPEC_AUXILIARY_DATA
    alpha, lumi = point["bkg_norm"], point["lumi"]
    gamma_sr = np.array([point["stat_SR_bin_0"], point["stat_SR_bin_1"]])
    gamma_cr = np.array([point["sys_CR_bin_0"], point["sys_CR_bin_2"]])
    sigma_sr = np.array([5.0 / 50.0, 4.0 / 60.0])
    tau = observed["constraint_sys_CR"]
    delta = norm.logpdf(swapped["constraint_bkg_norm"], alpha, 1.0) - norm.logpdf(
        0.0, alpha, 1.0
    )
    delta += np.sum(
        norm.logpdf(swapped["constraint_bblite_model_SR"], gamma_sr, sigma_sr)
        - norm.logpdf(1.0, gamma_sr, sigma_sr)
    )
    delta += np.sum(
        log_poisson(swapped["constraint_sys_CR"], gamma_cr * tau)
        - log_poisson(tau, gamma_cr * tau)
    )
    delta += norm.logpdf(swapped["lumi_constraint_meas"], lumi, 0.017) - norm.logpdf(
        1.02, lumi, 0.017
    )
    return float(delta)


@pytest.fixture(scope="module")
def model():
    return spec_model()


@pytest.fixture(scope="module")
def points(model):
    """Two parameter vectors of the compiled NLL, shape (P,) each."""
    nll = model.compile_nll()
    other = nll.initial.copy()
    other[:] = np.linspace(0.8, 1.3, len(other))
    return [nll.initial, other]


class TestCompiledAuxiliaryData:
    def test_default_is_the_observed_auxiliary_data(self, model, points):
        baked = model.compile_nll()
        swappable = model.compile_nll(auxdata_inputs=True)

        for x in points:
            value, grad = swappable.value_and_grad(x)
            assert value == pytest.approx(baked.value(x), rel=1e-12)
            assert_allclose(grad, baked.grad(x), rtol=1e-10)
            assert swappable.value(x, data=model.auxiliary_data) == value

    def test_swapped_auxiliary_data_change_only_the_constraints(self, model, points):
        swappable = model.compile_nll(auxdata_inputs=True)

        for x in points:
            expected = swappable.value(x) - constraint_log_delta(swappable.to_dict(x))
            assert swappable.value(x, data=SWAPPED_AUXILIARY_DATA) == pytest.approx(
                expected, rel=1e-12
            )

    def test_matches_a_model_built_with_that_lumi_auxdata(self, model, points):
        swappable = model.compile_nll(auxdata_inputs=True)
        reference = spec_model_with_lumi_auxdata(0.99).compile_nll()
        data = {**model.auxiliary_data, "lumi_constraint_meas": np.array(0.99)}

        for x in points:
            value, grad = swappable.value_and_grad(x, data=data)
            assert value == pytest.approx(reference.value(x), rel=1e-12)
            assert_allclose(grad, reference.grad(x), rtol=1e-10)
            assert_allclose(
                swappable.hessian(x, data=data), reference.hessian(x), rtol=1e-9
            )

    def test_with_observed_counts(self, model, points):
        """With both flags, one data mapping holds the counts and the auxiliary data."""
        counts_only = model.compile_nll(data_inputs=True)
        both = model.compile_nll(data_inputs=True, auxdata_inputs=True)

        for x in points:
            expected = counts_only.value(x, data=SWAPPED_COUNTS) - constraint_log_delta(
                both.to_dict(x)
            )
            value = both.value(x, data={**SWAPPED_COUNTS, **SWAPPED_AUXILIARY_DATA})
            assert value == pytest.approx(expected, rel=1e-12)
            assert both.value(x) == pytest.approx(counts_only.value(x), rel=1e-12)

    def test_switching_back_to_the_default(self, model, points):
        swappable = model.compile_nll(auxdata_inputs=True)
        x = points[1]
        default = swappable.value(x)
        default_hessian = swappable.hessian(x)

        assert swappable.value(x, data=SWAPPED_AUXILIARY_DATA) != pytest.approx(default)
        assert not np.allclose(
            swappable.hessian(x, data=SWAPPED_AUXILIARY_DATA), default_hessian
        )
        assert swappable.value(x) == default
        assert_allclose(swappable.hessian(x), default_hessian, rtol=0)

    def test_cache_entry_is_shared_across_auxiliary_data(self, tmp_path, points):
        """The cached function does not hold the auxiliary data; each model passes its own."""
        first = spec_model().compile_nll(auxdata_inputs=True, cache=tmp_path)
        other = spec_model_with_lumi_auxdata(0.99)
        second = other.compile_nll(auxdata_inputs=True, cache=tmp_path)

        assert (first.from_cache, second.from_cache) == (False, True)
        x = points[1]
        assert second.value(x) == pytest.approx(other.compile_nll().value(x), rel=1e-12)

    def test_cache_entries_differ_by_inputs(self, tmp_path):
        spec_model().compile_nll(auxdata_inputs=True, cache=tmp_path)
        counts = spec_model().compile_nll(data_inputs=True, cache=tmp_path)
        both = spec_model().compile_nll(
            data_inputs=True, auxdata_inputs=True, cache=tmp_path
        )
        assert (counts.from_cache, both.from_cache) == (False, False)


class TestCompiledAuxiliaryDataErrors:
    def test_data_needs_the_inputs(self, model, points):
        nll = model.compile_nll()
        with pytest.raises(ValueError, match="auxdata_inputs=True"):
            nll.value(points[0], data=SWAPPED_AUXILIARY_DATA)

    def test_counts_without_data_inputs(self, model, points):
        nll = model.compile_nll(auxdata_inputs=True)
        with pytest.raises(KeyError, match="model_SR"):
            nll.value(points[0], data={**SWAPPED_AUXILIARY_DATA, **SWAPPED_COUNTS})

    def test_missing_auxiliary_datum(self, model, points):
        nll = model.compile_nll(data_inputs=True, auxdata_inputs=True)
        data = {**SWAPPED_COUNTS, **SWAPPED_AUXILIARY_DATA}
        del data["constraint_sys_CR"]
        with pytest.raises(KeyError, match="constraint_sys_CR"):
            nll.value(points[0], data=data)

    @pytest.mark.parametrize(
        ("name", "values", "shape"),
        [
            pytest.param(
                "constraint_sys_CR", np.array([1.0, 2.0, 3.0]), r"\(2,\)", id="bins"
            ),
            pytest.param("constraint_bkg_norm", np.array([0.0]), r"\(\)", id="scalar"),
        ],
    )
    def test_wrong_shape(self, model, points, name, values, shape):
        nll = model.compile_nll(auxdata_inputs=True)
        with pytest.raises(ValueError, match=rf"{name}.*{shape}"):
            nll.value(points[0], data={**SWAPPED_AUXILIARY_DATA, name: values})

    def test_no_auxiliary_data(self):
        spec = copy.deepcopy(SPEC)
        for channel in spec["channels"]:
            for sample in channel["samples"]:
                sample["modifiers"] = [
                    m for m in sample["modifiers"] if m["type"] == "normfactor"
                ]
        spec["measurements"][0]["config"]["parameters"] = []
        ws = pyhs3.from_hifa(spec)
        model = ws.model(ws.analyses["meas"], progress=False)
        with pytest.raises(ValueError, match="auxiliary data"):
            model.compile_nll(auxdata_inputs=True)


def test_cache_key_tells_unnamed_global_observables_apart(tmp_path):
    """Two literal observables of the same type, keyed the other way round."""

    def two_constraints(first: str, second: str) -> pyhs3.Model:
        aux = [
            {
                "name": first,
                "type": "gaussian_dist",
                "x": 0.0,
                "mean": "alpha",
                "sigma": 1.0,
            },
            {
                "name": second,
                "type": "gaussian_dist",
                "x": 0.0,
                "mean": "beta",
                "sigma": 2.0,
            },
        ]
        return hs3_model(
            [normsys("Gauss")], aux=aux, parameters={"alpha": 0.0, "beta": 0.0}
        )

    first = two_constraints("c1", "c2")
    second = two_constraints("c2", "c1")
    first.compile_nll(auxdata_inputs=True, cache=tmp_path)
    nll = second.compile_nll(auxdata_inputs=True, cache=tmp_path)
    reference = second.compile_nll(auxdata_inputs=True)

    data = {**second.auxiliary_data, "c1": np.array(1.5), "c2": np.array(-0.5)}
    x = nll.initial
    assert nll.value(x, data=data) == pytest.approx(
        reference.value(x, data=data), rel=1e-12
    )


# A point of SPEC away from the nominal values of its constrained parameters.
POINT = {
    "mu": 1.2,
    "bkg_norm": 0.3,
    "lumi": 1.05,
    "stat_SR_bin_0": 1.05,
    "stat_SR_bin_1": 0.97,
    "sys_CR_bin_0": 1.1,
    "sys_CR_bin_2": 0.92,
}

N_TOYS = 20_000


def assert_moments(draws: np.ndarray, mean: Any, variance: Any) -> None:
    """Mean within 5 standard errors and variance within 5% of the expected ones."""
    stderr = np.sqrt(np.asarray(variance) / draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 5 * stderr)
    assert_allclose(draws.var(axis=0), variance, rtol=0.05)


class TestSampleAuxiliaryData:
    def test_keys_and_shapes(self, model):
        toys = model.sample_data(
            4, rng=np.random.default_rng(0), include_auxdata=True, **POINT
        )
        assert set(toys) == set(model.observed_data) | set(SPEC_AUXILIARY_DATA)
        assert toys["model_SR"].shape == (4, 2)
        assert toys["constraint_bkg_norm"].shape == (4,)
        assert toys["lumi_constraint_meas"].shape == (4,)
        assert toys["constraint_bblite_model_SR"].shape == (4, 2)
        assert toys["constraint_sys_CR"].shape == (4, 2)
        for draws in toys.values():
            assert draws.dtype == np.float64

    def test_main_measurement_only_by_default(self, model):
        toys = model.sample_data(4, rng=np.random.default_rng(0), **POINT)
        assert list(toys) == list(model.observed_data)

    def test_main_draws_come_first(self, model):
        """The counts of a full toy are those of a main-measurement toy of the same seed."""
        full = model.sample_data(
            3, rng=np.random.default_rng(4), include_auxdata=True, **POINT
        )
        main = model.sample_data(3, rng=np.random.default_rng(4), **POINT)
        for name, counts in main.items():
            np.testing.assert_array_equal(full[name], counts)

    def test_seeded_reproducibility(self, model):
        first = model.sample_data(
            5, rng=np.random.default_rng(42), include_auxdata=True, **POINT
        )
        second = model.sample_data(
            5, rng=np.random.default_rng(42), include_auxdata=True, **POINT
        )
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_moments(self, model):
        toys = model.sample_data(
            N_TOYS, rng=np.random.default_rng(1), include_auxdata=True, **POINT
        )
        assert_moments(toys["constraint_bkg_norm"], POINT["bkg_norm"], 1.0)
        assert_moments(toys["lumi_constraint_meas"], POINT["lumi"], 0.017**2)
        gamma_sr = np.array([POINT["stat_SR_bin_0"], POINT["stat_SR_bin_1"]])
        sigma_sr = np.array([5.0 / 50.0, 4.0 / 60.0])
        assert_moments(toys["constraint_bblite_model_SR"], gamma_sr, sigma_sr**2)
        # Poisson(gamma * tau): whole-number draws whose variance is the mean.
        rate = np.array([POINT["sys_CR_bin_0"], POINT["sys_CR_bin_2"]]) * [100.0, 144.0]
        assert_moments(toys["constraint_sys_CR"], rate, rate)
        np.testing.assert_array_equal(
            toys["constraint_sys_CR"], np.round(toys["constraint_sys_CR"])
        )

    def test_toys_feed_compiled_nll(self, model):
        nll = model.compile_nll(data_inputs=True, auxdata_inputs=True)
        x = nll.to_vector({name: POINT[name] for name in nll.parameter_names})
        toys = model.sample_data(
            3, rng=np.random.default_rng(2), include_auxdata=True, **POINT
        )
        for i in range(3):
            toy = {name: draws[i] for name, draws in toys.items()}
            assert np.isfinite(nll.value(x, data=toy))

    def test_compiles_once(self, monkeypatch):
        model = spec_model()
        calls = []
        compile_function = model_module.function

        def counting_function(*args, **kwargs):
            calls.append(kwargs.get("name"))
            return compile_function(*args, **kwargs)

        monkeypatch.setattr(model_module, "function", counting_function)
        for _ in range(2):
            model.sample_data(
                2, rng=np.random.default_rng(0), include_auxdata=True, **POINT
            )
        assert calls == ["expected_data", "auxiliary_parameters"]

    def test_missing_parameters(self, model):
        point = {k: v for k, v in POINT.items() if k != "bkg_norm"}
        with pytest.raises(ValueError, match="bkg_norm"):
            model.sample_data(
                2, rng=np.random.default_rng(0), include_auxdata=True, **point
            )


class TestSampleConstraintTypes:
    @pytest.mark.parametrize(
        ("constraint", "mean", "variance"),
        # alpha = 0.5: Gauss(alpha, 1), Poisson(alpha), LogNormal(mu = alpha, 1).
        [
            ("Gauss", 0.5, 1.0),
            ("Poisson", 0.5, 0.5),
            ("LogNormal", np.exp(1.0), (np.e - 1.0) * np.exp(2.0)),
        ],
    )
    def test_single_parameter(self, constraint, mean, variance):
        model = hs3_model([normsys(constraint)], parameters={"alpha": 0.5})
        toys = model.sample_data(
            N_TOYS,
            rng=np.random.default_rng(3),
            include_auxdata=True,
            mu=1.0,
            alpha=0.5,
        )
        draws = toys["constraint_alpha"]
        if constraint == "LogNormal":
            # The log of the draws is Gauss(alpha, 1); the draws themselves
            # are too heavy-tailed for a 5% variance check.
            assert_moments(np.log(draws), 0.5, 1.0)
        else:
            assert_moments(draws, mean, variance)

    def test_shapesys_lognormal(self):
        model = hs3_model([shapesys("LogNormal")], parameters=GAMMAS)
        toys = model.sample_data(
            N_TOYS,
            rng=np.random.default_rng(5),
            include_auxdata=True,
            mu=1.0,
            gamma_0=1.2,
            gamma_1=0.9,
        )
        # Median gamma, log-scale width sigma_rel = (0.2, 0.05).
        assert_moments(
            np.log(toys["constraint_sys"]), np.log([1.2, 0.9]), [0.2**2, 0.05**2]
        )

    def test_staterror_full_poisson(self):
        model = hs3_model(
            [staterror("Poisson", full=True)],
            parameters=GAMMAS,
            barlow_beeston_method="full",
        )
        toys = model.sample_data(
            N_TOYS,
            rng=np.random.default_rng(6),
            include_auxdata=True,
            mu=1.0,
            gamma_0=1.1,
            gamma_1=0.95,
        )
        rate = np.array([1.1, 0.95]) * [100.0, 400.0]
        assert_moments(toys["constraint_stat"], rate, rate)

    def test_root_style_gaussian(self):
        """Gauss(alpha | nom_alpha, sigma): nom_alpha is drawn around alpha."""
        aux = {
            "name": "c",
            "type": "gaussian_dist",
            "x": "alpha",
            "mean": "nom_alpha",
            "sigma": 0.5,
        }
        model = hs3_model(
            [normsys("Gauss")],
            aux=[aux],
            parameters={"alpha": 0.0},
            const={"nom_alpha": 0.0},
        )
        toys = model.sample_data(
            N_TOYS,
            rng=np.random.default_rng(7),
            include_auxdata=True,
            mu=1.0,
            alpha=-0.4,
        )
        assert_moments(toys["c"], -0.4, 0.25)


class TestSampleErrors:
    @pytest.mark.parametrize(
        ("aux", "parameters", "const", "named"),
        [
            pytest.param(
                {"name": "c", "type": "exponential_dist", "x": 1.0, "c": "alpha"},
                {"alpha": -1.0},
                {},
                "'c' (ExponentialDist)",
                id="type",
            ),
            pytest.param(
                {
                    "name": "c",
                    "type": "gaussian_dist",
                    "x": "alpha",
                    "mean": "beta",
                    "sigma": 1.0,
                },
                {"alpha": 0.0, "beta": 0.0},
                {},
                "'c' (GaussianDist)",
                id="no_global_observable",
            ),
        ],
    )
    def test_unsampleable_term(self, aux, parameters, const, named):
        model = hs3_model(
            [normsys("Gauss")], aux=[aux], parameters=parameters, const=const
        )
        with pytest.raises(NotImplementedError, match=re.escape(named)):
            model.sample_data(
                2,
                rng=np.random.default_rng(0),
                include_auxdata=True,
                mu=1.0,
                **parameters,
            )
        # Toys of the main measurement alone stay available.
        model.sample_data(2, rng=np.random.default_rng(0), mu=1.0, **parameters)

    def test_non_finite_parameter(self):
        # gamma_0 = 0 gives a LogNormal log-scale location of log(0) = -inf.
        model = hs3_model([shapesys("LogNormal")], parameters=GAMMAS)
        with pytest.raises(ValueError, match="constraint_sys"):
            model.sample_data(
                2,
                rng=np.random.default_rng(0),
                include_auxdata=True,
                mu=1.0,
                gamma_0=0.0,
                gamma_1=1.0,
            )

    def test_negative_poisson_rate(self):
        model = hs3_model([normsys("Poisson")], parameters={"alpha": 0.5})
        with pytest.raises(ValueError, match="constraint_alpha"):
            model.sample_data(
                2,
                rng=np.random.default_rng(0),
                include_auxdata=True,
                mu=1.0,
                alpha=-0.5,
            )
