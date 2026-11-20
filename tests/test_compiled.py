"""Tests for Model.compile_nll and CompiledNLL."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import minimize
from scipy.stats import truncnorm

from pyhs3 import Workspace
from pyhs3.compiled import CompiledNLL

_EVENTS = [0.4, 1.1, 1.9, 2.3, 2.8, 3.6]


def _workspace(extra_parameters: list[dict] | None = None) -> Workspace:
    """Unbinned Gaussian with free mean and width on [-10, 10]."""
    return Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "g",
                "type": "gaussian_dist",
                "x": "x_obs",
                "mean": "mu",
                "sigma": "sigma",
            }
        ],
        domains=[
            {
                "name": "main",
                "type": "product_domain",
                "axes": [
                    {"name": "mu", "min": -5.0, "max": 5.0},
                    {"name": "sigma", "min": 0.1, "max": 10.0},
                ],
            }
        ],
        data=[
            {
                "name": "d",
                "type": "unbinned",
                "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
                "entries": [[x] for x in _EVENTS],
            }
        ],
        likelihoods=[{"name": "L", "distributions": ["g"], "data": ["d"]}],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
        parameter_points=[
            {
                "name": "init",
                "parameters": [
                    {"name": "mu", "value": 1.0},
                    {"name": "sigma", "value": 2.0},
                    *(extra_parameters or []),
                ],
            }
        ],
    )


def _reference_nll(mu: float, sigma: float) -> float:
    a, b = (-10.0 - mu) / sigma, (10.0 - mu) / sigma
    return -float(np.sum(truncnorm.logpdf(_EVENTS, a, b, loc=mu, scale=sigma)))


@pytest.fixture
def model():
    ws = _workspace()
    return ws.model(ws.analyses["A"], progress=False)


class TestCompiledNLLInterface:
    def test_returns_compiled_nll(self, model):
        assert isinstance(model.compile_nll(), CompiledNLL)

    def test_parameters_initial_and_bounds(self, model):
        """Free parameters come in a fixed order with initial values and bounds."""
        nll = model.compile_nll()

        assert set(nll.parameter_names) == {"mu", "sigma"}
        point = dict(zip(nll.parameter_names, nll.initial, strict=True))
        assert point == {"mu": 1.0, "sigma": 2.0}
        bounds = dict(zip(nll.parameter_names, nll.bounds, strict=True))
        assert bounds == {"mu": (-5.0, 5.0), "sigma": (0.1, 10.0)}
        assert nll.initial.shape == (2,)

    def test_vector_and_dict_round_trip(self, model):
        nll = model.compile_nll()
        point = {"mu": 0.3, "sigma": 1.7}

        x = nll.to_vector(point)

        assert x.shape == (2,)
        assert nll.to_dict(x) == point

    def test_to_vector_rejects_unknown_or_missing_names(self, model):
        nll = model.compile_nll()
        with pytest.raises(KeyError, match="sigma"):
            nll.to_vector({"mu": 0.3})
        with pytest.raises(KeyError, match="nope"):
            nll.to_vector({"mu": 0.3, "sigma": 1.0, "nope": 2.0})

    def test_wrong_length_vector_raises(self, model):
        nll = model.compile_nll()
        with pytest.raises(ValueError, match="expected 2"):
            nll.value(np.array([1.0, 2.0, 3.0]))


class TestCompiledNLLValues:
    def test_value_matches_reference(self, model):
        """value(x) is the NLL -sum(log_prob) with data baked in."""
        nll = model.compile_nll()
        x = nll.to_vector({"mu": 0.7, "sigma": 1.4})

        assert nll.value(x) == pytest.approx(_reference_nll(0.7, 1.4), rel=1e-8)

    def test_value_matches_log_prob(self, model):
        nll = model.compile_nll()
        inputs = [v for v in explicit_graph_inputs([model.log_prob]) if v.name]
        log_prob = pytensor.function(inputs, model.log_prob)
        point = {"mu": -0.4, "sigma": 2.5}

        expected = -float(np.sum(log_prob(**{**model.data, **point})))

        assert nll.value(nll.to_vector(point)) == pytest.approx(expected, rel=1e-12)

    def test_value_and_grad_matches_finite_differences(self, model):
        nll = model.compile_nll()
        x = nll.to_vector({"mu": 0.7, "sigma": 1.4})
        step = 1e-6

        value, grad = nll.value_and_grad(x)

        assert value == pytest.approx(nll.value(x), rel=1e-14)
        assert grad.shape == (2,)
        for i in range(2):
            up, down = x.copy(), x.copy()
            up[i] += step
            down[i] -= step
            fd = (nll.value(up) - nll.value(down)) / (2 * step)
            assert grad[i] == pytest.approx(fd, rel=1e-6)

    def test_grad_matches_value_and_grad(self, model):
        nll = model.compile_nll()
        x = nll.to_vector({"mu": 0.2, "sigma": 3.0})
        np.testing.assert_array_equal(nll.grad(x), nll.value_and_grad(x)[1])

    def test_without_gradient(self, model):
        nll = model.compile_nll(gradient=False)
        x = nll.initial
        assert np.isfinite(nll.value(x))
        with pytest.raises(RuntimeError, match="gradient=False"):
            nll.value_and_grad(x)


class TestCompiledNLLErrors:
    def test_batched_free_parameter_is_rejected(self):
        ws = _workspace()
        ws.parameter_points["init"]["mu"].kind = pt.vector
        with pytest.warns(UserWarning, match="kind override vector"):
            model = ws.model(ws.analyses["A"], progress=False)
        with pytest.raises(ValueError, match="mu"):
            model.compile_nll()

    def test_requires_a_likelihood(self):
        ws = _workspace()
        model = ws.model(0, progress=False)
        with pytest.raises(RuntimeError, match="likelihood"):
            model.compile_nll()


class TestCompiledNLLCache:
    def test_cache_round_trip(self, tmp_path):
        ws = _workspace()
        first = ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)
        assert not first.from_cache
        assert len(list(tmp_path.iterdir())) == 1

        second = ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)

        assert second.from_cache
        x = first.to_vector({"mu": 0.7, "sigma": 1.4})
        assert second.value(x) == first.value(x)
        np.testing.assert_array_equal(second.grad(x), first.grad(x))

    def test_cache_key_depends_on_options_and_model(self, tmp_path):
        ws = _workspace()
        ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)
        ws.model(ws.analyses["A"], progress=False).compile_nll(
            cache=tmp_path, gradient=False
        )
        other = _workspace()
        other.parameter_points["init"]["sigma"].const = True
        other.model(other.analyses["A"], progress=False).compile_nll(cache=tmp_path)

        assert len(list(tmp_path.iterdir())) == 3


class TestCompiledNLLFit:
    def test_scipy_fit_recovers_the_maximum_likelihood_estimate(self, model):
        """End to end: L-BFGS-B with the analytic gradient finds the MLE."""
        nll = model.compile_nll()

        result = minimize(
            nll.value_and_grad,
            nll.initial,
            jac=True,
            bounds=nll.bounds,
            method="L-BFGS-B",
        )

        assert result.success
        fitted = nll.to_dict(result.x)
        # The data sit far inside [-10, 10], so the truncated-Gaussian MLE is
        # the sample mean and the (biased) sample standard deviation.
        assert fitted["mu"] == pytest.approx(np.mean(_EVENTS), abs=1e-4)
        assert fitted["sigma"] == pytest.approx(np.std(_EVENTS), abs=1e-4)
