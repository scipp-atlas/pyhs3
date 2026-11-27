"""Tests for SampleableDistribution: drawing a distribution's variate.

The Gaussian, Poisson and log-normal distributions draw their ``x`` given the
other fields; the Gaussian, symmetric in ``x`` and ``mean``, also draws its
``mean`` given ``x``.  Model.sample_data draws the global observables of the
constraint terms with them.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyhs3.context import Context
from pyhs3.distributions import (
    ExponentialDist,
    GaussianDist,
    LogNormalDist,
    PoissonDist,
    SampleableDistribution,
)

N = 200_000


def _context(**values: float | list[float]) -> Context:
    return Context(
        {
            name: pt.constant(np.asarray(v, dtype=np.float64))
            for name, v in values.items()
        }
    )


def _evaluate(parameters):
    return {name: np.asarray(tensor.eval()) for name, tensor in parameters.items()}


class TestSampleableTypes:
    @pytest.mark.parametrize("cls", [GaussianDist, PoissonDist, LogNormalDist])
    def test_constraint_types_are_sampleable(self, cls):
        assert issubclass(cls, SampleableDistribution)

    def test_exponential_is_not(self):
        assert not issubclass(ExponentialDist, SampleableDistribution)

    def test_sampled_fields(self):
        assert GaussianDist.sampled_fields == ("x", "mean")
        assert PoissonDist.sampled_fields == ("x",)
        assert LogNormalDist.sampled_fields == ("x",)


class TestGaussian:
    def test_field_tensor(self):
        dist = GaussianDist(name="g", x="obs", mean="mu", sigma="s")
        context = _context(obs=0.5, mu=1.0, s=2.0)
        assert dist.field_tensor(context, "x") is context["obs"]
        assert dist.field_tensor(context, "mean") is context["mu"]

    def test_draws_x_around_mean(self):
        dist = GaussianDist(name="g", x="obs", mean="mu", sigma="s")
        parameters = _evaluate(
            dist.sampling_parameters(_context(obs=0.0, mu=1.5, s=0.5), "x")
        )
        assert_allclose(parameters["mean"], 1.5)
        assert_allclose(parameters["sigma"], 0.5)

        draws = dist.sample(np.random.default_rng(0), parameters, (N,))

        assert draws.shape == (N,)
        assert draws.mean() == pytest.approx(1.5, abs=5 * 0.5 / np.sqrt(N))
        assert draws.std() == pytest.approx(0.5, rel=0.01)

    def test_draws_mean_around_x(self):
        """Gauss(x | mean, sigma) is symmetric in x and mean."""
        dist = GaussianDist(name="g", x="alpha", mean="nom_alpha", sigma=1.0)
        context = _context(alpha=-0.7, nom_alpha=0.0)
        parameters = _evaluate(
            dist.sampling_parameters(Context({**context, **dist.constants}), "mean")
        )
        assert_allclose(parameters["mean"], -0.7)
        assert_allclose(parameters["sigma"], 1.0)

    def test_per_bin_parameters_broadcast(self):
        """(B,) parameters and size (n, B) give one row of B draws per toy."""
        dist = GaussianDist(name="g", x="obs", mean="mu", sigma="s")
        parameters = {
            "mean": np.array([1.0, 1.0, 1.0]),
            "sigma": np.array([0.1, 0.2, 0.3]),
        }

        draws = dist.sample(np.random.default_rng(1), parameters, (N, 3))

        assert draws.shape == (N, 3)
        assert_allclose(draws.std(axis=0), [0.1, 0.2, 0.3], rtol=0.01)

    def test_unknown_field(self):
        dist = GaussianDist(name="g", x="obs", mean="mu", sigma="s")
        with pytest.raises(ValueError, match="sigma"):
            dist.sampling_parameters(_context(obs=0.0, mu=0.0, s=1.0), "sigma")


class TestPoisson:
    def test_draws_counts(self):
        dist = PoissonDist(name="p", x="n", mean="lam")
        parameters = _evaluate(dist.sampling_parameters(_context(n=4.0, lam=4.0), "x"))
        assert_allclose(parameters["mean"], 4.0)

        draws = dist.sample(np.random.default_rng(2), parameters, (N,))

        assert draws.dtype == np.float64
        assert_array_equal(draws, np.round(draws))
        assert draws.mean() == pytest.approx(4.0, abs=5 * 2.0 / np.sqrt(N))
        assert draws.var() == pytest.approx(4.0, rel=0.02)

    def test_unknown_field(self):
        dist = PoissonDist(name="p", x="n", mean="lam")
        with pytest.raises(ValueError, match="mean"):
            dist.sampling_parameters(_context(n=4.0, lam=4.0), "mean")


class TestLogNormal:
    def test_draws_x(self):
        dist = LogNormalDist(name="l", x="obs", mu="m", sigma="s")
        parameters = _evaluate(
            dist.sampling_parameters(_context(obs=1.0, m=0.2, s=0.3), "x")
        )
        assert_allclose(parameters["mu"], 0.2)
        assert_allclose(parameters["sigma"], 0.3)

        draws = dist.sample(np.random.default_rng(3), parameters, (N,))

        assert np.log(draws).mean() == pytest.approx(0.2, abs=5 * 0.3 / np.sqrt(N))
        assert np.log(draws).std() == pytest.approx(0.3, rel=0.01)

    def test_unknown_field(self):
        dist = LogNormalDist(name="l", x="obs", mu="m", sigma="s")
        with pytest.raises(ValueError, match="mu"):
            dist.sampling_parameters(_context(obs=1.0, m=0.2, s=0.3), "mu")


def test_same_seed_same_draws():
    dist = GaussianDist(name="g", x="obs", mean="mu", sigma="s")
    parameters = {"mean": np.array(0.0), "sigma": np.array(1.0)}
    first = dist.sample(np.random.default_rng(5), parameters, (10,))
    second = dist.sample(np.random.default_rng(5), parameters, (10,))
    assert_array_equal(first, second)
