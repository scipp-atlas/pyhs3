"""Regression tests for log-space likelihood evaluation.

The probability-space HistFactory likelihood multiplies per-bin Poisson
probabilities (``pt.prod(pt.exp(log_probs))``).  For channels with many bins
and/or large expected counts this product underflows float64 to ``0.0``, so the
old ``logpdf = np.log(pdf(...))`` returned ``-inf`` even though every per-bin
log-probability is perfectly finite.

These tests assert that:

* ``model.pdf(...)`` underflows to ``0.0`` for such a channel,
* ``model.logpdf(...)`` / ``logpdf_unsafe(...)`` stay finite and match the
  analytic sum of per-bin Poisson log-pmfs,
* the likelihood ``log_prob`` path is finite too, and
* ``logpdf == log(pdf)`` for a simple (representable) Gaussian model.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

import pyhs3 as hs3
from pyhs3.context import Context
from pyhs3.data import BinnedData, Data
from pyhs3.distributions import Distributions, HistFactoryDistChannel
from pyhs3.distributions.basic import (
    ExponentialDist,
    GaussianDist,
    LogNormalDist,
    PoissonDist,
)
from pyhs3.domains import Domains, ProductDomain
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoints, ParameterSet
from pyhs3.typing.aliases import TensorVar
from pyhs3.workspace import Workspace

# scipy.stats is preferred for analytic references; fall back to closed forms
# built from stdlib math if scipy is unavailable in the test environment.
try:
    from scipy.stats import lognorm as _scipy_lognorm
    from scipy.stats import norm as _scipy_norm
    from scipy.stats import poisson as _scipy_poisson

    def _poisson_logpmf(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
        return np.asarray(_scipy_poisson.logpmf(counts, rates), dtype=np.float64)

    def _norm_logpdf(x: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.asarray(
            _scipy_norm.logpdf(x, loc=mean, scale=sigma), dtype=np.float64
        )

    def _lognorm_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.asarray(
            _scipy_lognorm.logpdf(x, s=sigma, scale=np.exp(mu)), dtype=np.float64
        )

except ModuleNotFoundError:  # pragma: no cover - scipy is in the dev deps

    def _poisson_logpmf(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
        # math.lgamma is a stdlib built-in (Python ≥ 3.5); no external deps needed.
        lgamma = np.vectorize(math.lgamma)
        return counts * np.log(rates) - rates - lgamma(counts + 1)

    def _norm_logpdf(x: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        z = (x - mean) / sigma
        return -0.5 * z**2 - np.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    def _lognorm_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        z = (np.log(x) - mu) / sigma
        return -np.log(x) - np.log(sigma) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z**2


def _underflow_workspace() -> tuple[Workspace, np.ndarray]:
    """Build a single-channel HFDC workspace whose probability underflows.

    Returns the workspace and the observed/expected bin counts (observed equals
    expected so the analytic Poisson log-pmf is straightforward).
    """
    nbins = 200
    rng = np.random.default_rng(0)
    # Thousands of counts per bin: each bin's Poisson pmf at its mode is ~1/sqrt(2*pi*rate)
    # (~5e-3 here); the product across this many bins underflows float64 to 0.0
    # (the smallest positive double is ~2.2e-308).
    contents = (5000.0 + rng.integers(0, 1000, size=nbins)).astype(float).tolist()

    obs_name = "x_SR"
    channel = HistFactoryDistChannel(
        name="SR",
        axes=[{"name": obs_name, "min": 0.0, "max": 10.0, "nbins": nbins}],
        samples=[
            {
                "name": "signal",
                "data": {"contents": contents, "errors": [1.0] * nbins},
                "modifiers": [],
            }
        ],
    )
    binned = BinnedData(
        name="SR_data",
        axes=[{"name": obs_name, "min": 0.0, "max": 10.0, "nbins": nbins}],
        contents=contents,
    )
    ws = Workspace(
        metadata=Metadata(hs3_version="0.3.0"),
        distributions=Distributions([channel]),
        data=Data([binned]),
        likelihoods=Likelihoods(
            [Likelihood(name="L", distributions=[channel], data=[binned])]
        ),
        domains=Domains([ProductDomain(name="default")]),
        parameter_points=ParameterPoints([ParameterSet(name="default", parameters=[])]),
    )
    return ws, np.asarray(contents, dtype=np.float64)


def test_pdf_underflows_but_logpdf_is_finite():
    """pdf underflows to 0.0; logpdf stays finite and matches Poisson log-pmf sum."""
    ws, contents = _underflow_workspace()
    model = ws.model(next(iter(ws.likelihoods)), progress=False)

    observed = np.asarray(contents)
    # observed == expected for every bin
    pdf_val = model.pdf_unsafe("SR", SR_observed=observed)
    logpdf_val = model.logpdf_unsafe("SR", SR_observed=observed)

    # Probability-space product underflows to (essentially) zero ...
    assert float(np.asarray(pdf_val)) == 0.0

    # ... but the log-space value is finite (the old np.log(pdf) gave -inf).
    logpdf_scalar = float(np.asarray(logpdf_val))
    assert math.isfinite(logpdf_scalar)

    # And it matches the analytic sum of per-bin Poisson log-pmfs.
    expected = float(np.sum(_poisson_logpmf(observed, observed)))
    assert logpdf_scalar == pytest.approx(expected, rel=1e-10)


def test_log_prob_is_finite_under_underflow():
    """The likelihood log_prob path is finite where the probability product underflows."""
    ws, contents = _underflow_workspace()
    model = ws.model(next(iter(ws.likelihoods)), progress=False)

    lp = model.log_prob
    inputs = {
        v.name: v
        for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
        if v.name
    }
    fn = pytensor.function(list(inputs.values()), lp)
    val = float(np.asarray(fn(**model.data, **model.nominal_params)).item())

    assert math.isfinite(val)
    observed = np.asarray(contents)
    expected = float(np.sum(_poisson_logpmf(observed, observed)))
    assert val == pytest.approx(expected, rel=1e-10)


def test_logpdf_matches_log_pdf_for_gaussian():
    """For a representable Gaussian model, logpdf equals log(pdf)."""
    ws = hs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": "sigma",
            }
        ],
        domains=[
            {
                "name": "default",
                "type": "product_domain",
                "axes": [{"name": "x", "min": -10.0, "max": 10.0}],
            }
        ],
        parameter_points=[
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                    {"name": "sigma", "value": 1.0},
                ],
            }
        ],
    )
    model = ws.model(0, progress=False)

    for x in (-2.0, 0.0, 1.5, 3.0):
        pdf_val = float(np.asarray(model.pdf_unsafe("gauss", x=x, mu=0.0, sigma=1.0)))
        logpdf_val = float(
            np.asarray(model.logpdf_unsafe("gauss", x=x, mu=0.0, sigma=1.0))
        )
        assert logpdf_val == pytest.approx(math.log(pdf_val), rel=1e-12)


def test_zero_bin_logprob_is_not_nan():
    """When observed==0 and expected==0, the bin's Poisson log-pmf must be 0 (not NaN).

    When the HFDC observed data is a symbolic pt.vector (built without a likelihood
    context so it is not baked as a constant), PyTensor evaluates
    ``0 * pt.log(0) = 0 * -inf = NaN``.  The correct Poisson log-pmf for k=0 is
    just ``-expected`` (i.e. 0 when expected==0 too).  A ``pt.switch`` guard fixes
    this: when observed==0, return only ``-expected_rates``.
    """
    channel = HistFactoryDistChannel(
        name="zero",
        axes=[{"name": "x_zero", "min": 0.0, "max": 1.0, "nbins": 1}],
        samples=[
            {
                "name": "signal",
                "data": {"contents": [5.0], "errors": [1.0]},
                "modifiers": [
                    # normfactor lets us drive expected_rates → 0 at mu=0
                    {"name": "mu", "type": "normfactor", "parameter": "mu"},
                ],
            }
        ],
    )
    ws = Workspace(
        metadata=Metadata(hs3_version="0.3.0"),
        distributions=Distributions([channel]),
        data=Data([]),
        likelihoods=Likelihoods([]),
        domains=Domains([ProductDomain(name="default")]),
        parameter_points=ParameterPoints(
            [
                ParameterSet(
                    name="default",
                    parameters=[{"name": "mu", "value": 0.0}],
                )
            ]
        ),
    )
    # Build without a likelihood so zero_observed stays symbolic (not baked).
    model = ws.model(0, progress=False)
    observed = np.zeros(1, dtype=np.float64)

    # logpdf at mu=0 → expected_rates=0.  With obs=0 this is Poisson(0|0)=1 → log=0.
    # Without the pt.switch guard, PyTensor computes 0*log(0) = NaN.
    logpdf_val = float(
        np.asarray(
            model.logpdf_unsafe("zero", zero_observed=observed, mu=np.float64(0.0))
        )
    )
    assert math.isfinite(logpdf_val), (
        f"logpdf should be finite (not NaN), got {logpdf_val}"
    )
    assert logpdf_val == pytest.approx(0.0, abs=1e-12)


def test_zero_bin_logprob_gradient_is_not_nan():
    """The gradient of the Poisson log-pmf wrt expected_rates must stay finite
    at observed==0, expected==0, independent of graph rewrites.

    ``_bin_log_probs`` guards the *value* of ``0 * log(0)`` with a ``pt.switch``
    on its output, but PyTensor differentiates every branch of a switch,
    including the one not selected at runtime. The untaken branch still
    contains ``observed * log(expected_rates)``, whose gradient wrt
    ``expected_rates`` is ``observed / expected_rates = 0 / 0 = NaN`` at this
    point; multiplying that NaN by the switch's zero mask does not clear it
    (``0 * NaN = NaN``). Compiling with ``optimizer=None`` disables the graph
    rewrites that otherwise happen to simplify this away, so the test exposes
    the NaN structurally rather than relying on optimizer behavior.
    """
    channel = HistFactoryDistChannel(
        name="grad_probe",
        axes=[{"name": "x_probe", "min": 0.0, "max": 2.0, "nbins": 2}],
        samples=[
            {
                "name": "signal",
                "data": {"contents": [1.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            }
        ],
    )
    observed = pt.constant(np.array([0.0, 2.0]))
    context = Context(parameters={"grad_probe_observed": observed})
    expected_rates = pt.vector("expected_rates")

    log_probs = channel._bin_log_probs(context, expected_rates)
    grad = pt.grad(pt.sum(log_probs), expected_rates)

    mode = pytensor.compile.mode.Mode(linker="py", optimizer=None)
    fn = pytensor.function([expected_rates], grad, mode=mode)

    grad_val = fn(np.array([0.0, 3.0]))
    assert np.all(np.isfinite(grad_val)), f"gradient has NaN/Inf: {grad_val}"


# --------------------------------------------------------------------------
# Layer 1 of #243: analytic log_likelihood() on GaussianDist, PoissonDist, and
# LogNormalDist, so neither of them has to round-trip through a probability
# value that can underflow to 0.0 before Layer 2 (modifier-level
# log_constraint(), tracked separately) can rely on it.
# --------------------------------------------------------------------------


def _c(value: float) -> TensorVar:
    """pt.constant() at explicit float64: bare Python floats otherwise get cast
    to float32, which is nowhere near enough precision for the rtol=1e-12
    parity checks below."""
    return cast(TensorVar, pt.constant(value, dtype="float64"))


class TestGaussianLogLikelihood:
    """GaussianDist.log_likelihood is the analytic log form of likelihood()."""

    @pytest.mark.parametrize(
        ("x", "mean", "sigma"),
        [
            pytest.param(0.0, 0.0, 1.0, id="standard_normal_at_mode"),
            pytest.param(2.0, 0.0, 1.0, id="standard_normal_offset"),
            pytest.param(130.0, 125.0, 10.0, id="hep_mass_peak"),
            pytest.param(-5.0, 3.0, 2.5, id="negative_x"),
        ],
    )
    def test_matches_log_of_likelihood(self, x, mean, sigma):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = GaussianDist(name="g", x="x", mean="mean", sigma="sigma")
        context = Context(
            {
                "x": _c(x),
                "mean": _c(mean),
                "sigma": _c(sigma),
            }
        )
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    @pytest.mark.parametrize(
        ("x", "mean", "sigma"),
        [
            pytest.param(0.0, 0.0, 1.0, id="standard_normal_at_mode"),
            pytest.param(2.0, 0.0, 1.0, id="standard_normal_offset"),
            pytest.param(130.0, 125.0, 10.0, id="hep_mass_peak"),
        ],
    )
    def test_matches_scipy_norm_logpdf(self, x, mean, sigma):
        """likelihood() is the normalized Gaussian pdf, so this matches scipy exactly."""
        dist = GaussianDist(name="g", x="x", mean="mean", sigma="sigma")
        context = Context(
            {
                "x": _c(x),
                "mean": _c(mean),
                "sigma": _c(sigma),
            }
        )
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        expected = float(_norm_logpdf(np.array(x), np.array(mean), np.array(sigma)))
        assert log_val == pytest.approx(expected, rel=1e-10)

    def test_finite_at_large_pull_while_likelihood_underflows(self):
        """|x-mu|/sigma = 40: likelihood underflows to 0.0, log_likelihood stays finite (~-800)."""
        dist = GaussianDist(name="g", x="x", mean="mean", sigma="sigma")
        context = Context(
            {
                "x": _c(40.0),
                "mean": _c(0.0),
                "sigma": _c(1.0),
            }
        )
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)
        expected = -0.5 * 40.0**2 - 0.5 * math.log(2.0 * math.pi)
        assert log_val == pytest.approx(expected, rel=1e-10)
        assert log_val == pytest.approx(-800.0, rel=1e-2)


class TestPoissonLogLikelihood:
    """PoissonDist.log_likelihood returns the analytic log-pmf directly."""

    @pytest.mark.parametrize(
        ("mean", "x"),
        [
            pytest.param(1.0, 0.0, id="lambda1_k0"),
            pytest.param(3.0, 3.0, id="at_mode"),
            pytest.param(5.0, 8.0, id="above_mode"),
        ],
    )
    def test_matches_log_of_likelihood(self, mean, x):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = PoissonDist(name="p", mean="mean", x="x")
        context = Context({"mean": _c(mean), "x": _c(x)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    @pytest.mark.parametrize(
        ("mean", "x"),
        [
            pytest.param(1.0, 0.0, id="lambda1_k0"),
            pytest.param(3.0, 3.0, id="at_mode"),
            pytest.param(5.0, 8.0, id="above_mode"),
        ],
    )
    def test_matches_scipy_poisson_logpmf(self, mean, x):
        """likelihood() is the exact pmf, so this matches scipy.stats.poisson exactly."""
        dist = PoissonDist(name="p", mean="mean", x="x")
        context = Context({"mean": _c(mean), "x": _c(x)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        expected = float(_poisson_logpmf(np.array(x), np.array(mean)))
        assert log_val == pytest.approx(expected, rel=1e-10)

    def test_stable_at_large_mean_and_count(self):
        """mean=x=1e5: the log-pmf subtracts ~1e6-magnitude terms to an O(1) result;
        the analytic form stays accurate despite that cancellation."""
        dist = PoissonDist(name="p", mean="mean", x="x")
        context = Context({"mean": _c(1e5), "x": _c(1e5)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())

        assert math.isfinite(log_val)
        expected = float(_poisson_logpmf(np.array(1e5), np.array(1e5)))
        assert log_val == pytest.approx(expected, rel=1e-8)
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-10)

    def test_finite_at_deep_tail_while_likelihood_underflows(self):
        """mean=100, x=1000 is far enough into the tail that the pmf underflows to 0.0."""
        dist = PoissonDist(name="p", mean="mean", x="x")
        context = Context({"mean": _c(100.0), "x": _c(1000.0)})
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)
        expected = float(_poisson_logpmf(np.array(1000.0), np.array(100.0)))
        assert log_val == pytest.approx(expected, rel=1e-8)

    def test_zero_zero_guard_matches_switch_pattern(self):
        """mean=0, x=0: Poisson(0|0)=1, so the log-pmf is 0 (not NaN from 0 * log(0))."""
        dist = PoissonDist(name="p", mean="mean", x="x")
        context = Context({"mean": _c(0.0), "x": _c(0.0)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        assert log_val == pytest.approx(0.0, abs=1e-12)


class TestLogNormalLogLikelihood:
    """LogNormalDist.log_likelihood is the analytic log form of likelihood()."""

    @pytest.mark.parametrize(
        ("x", "mu", "sigma"),
        [
            pytest.param(1.0, 0.0, 1.0, id="standard_at_one"),
            pytest.param(5.0, 1.0, 0.5, id="offset"),
            pytest.param(0.1, -1.0, 2.0, id="small_x"),
        ],
    )
    def test_matches_log_of_likelihood(self, x, mu, sigma):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = LogNormalDist(name="ln", x="x", mu="mu", sigma="sigma")
        context = Context({"x": _c(x), "mu": _c(mu), "sigma": _c(sigma)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    @pytest.mark.parametrize(
        ("x", "mu", "sigma"),
        [
            pytest.param(1.0, 0.0, 1.0, id="standard_at_one"),
            pytest.param(5.0, 1.0, 0.5, id="offset"),
            pytest.param(0.1, -1.0, 2.0, id="small_x"),
        ],
    )
    def test_matches_scipy_lognorm_logpdf(self, x, mu, sigma):
        """likelihood() is scipy's lognorm pdf with s=sigma, scale=exp(mu)."""
        dist = LogNormalDist(name="ln", x="x", mu="mu", sigma="sigma")
        context = Context({"x": _c(x), "mu": _c(mu), "sigma": _c(sigma)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        expected = float(_lognorm_logpdf(np.array(x), np.array(mu), np.array(sigma)))
        assert log_val == pytest.approx(expected, rel=1e-10)

    def test_finite_at_large_pull_while_likelihood_underflows(self):
        """z = (ln(x)-mu)/sigma = 40: likelihood underflows to 0.0, log_likelihood stays finite."""
        dist = LogNormalDist(name="ln", x="x", mu="mu", sigma="sigma")
        x_val = math.exp(40.0)
        context = Context(
            {
                "x": _c(x_val),
                "mu": _c(0.0),
                "sigma": _c(1.0),
            }
        )
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)
        expected = -math.log(x_val) - 0.5 * math.log(2.0 * math.pi) - 0.5 * 40.0**2
        assert log_val == pytest.approx(expected, rel=1e-10)


class TestBaseDistributionLogLikelihoodDefault:
    """A distribution with no analytic override falls back to pt.log(likelihood())."""

    @pytest.mark.parametrize(
        ("x", "c"),
        [
            pytest.param(1.0, 0.5, id="ordinary_point"),
            pytest.param(3.0, 2.0, id="larger_rate"),
        ],
    )
    def test_exponential_dist_default_log_likelihood(self, x, c):
        """ExponentialDist has no log_likelihood override, so it uses the base default."""
        dist = ExponentialDist(name="e", x="x", c="c")
        context = Context({"x": _c(x), "c": _c(c)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        assert log_val == pytest.approx(np.log(prob_val), rel=1e-12)


class TestLogExpressionUsesAnalyticLogLikelihood:
    """log_expression() is built from log_likelihood(), subtracting log(normalization)."""

    def test_without_observable_equals_log_likelihood(self):
        """No matching observable: no normalization applies, so log_expression == log_likelihood."""
        dist = GaussianDist(name="constraint", mean="nom", sigma="sigma", x="alpha")
        context = Context(
            parameters={
                "alpha": _c(0.5),
                "nom": _c(0.0),
                "sigma": _c(1.0),
            },
        )
        log_expr_val = float(pytensor.function([], dist.log_expression(context))())
        log_like_val = float(pytensor.function([], dist.log_likelihood(context))())
        assert log_expr_val == pytest.approx(log_like_val, rel=1e-12)

    def test_with_observable_matches_log_of_normalized_expression(self):
        """Observable present: log_expression == log(likelihood / normalization_integral).

        The observable leaf must be a pt.vector, not a pt.constant scalar: the
        Gauss-Legendre quadrature fallback substitutes it with a 64-node vector
        via graph_replace(), which every other normalization test in this
        codebase also does (see test_normalization.py).
        """
        dist = GaussianDist(name="gauss", mean="mu", sigma="sigma", x="x")
        x_var = pt.vector("x")
        context = Context(
            parameters={
                "x": x_var,
                "mu": _c(125.0),
                "sigma": _c(10.0),
            },
            observables={"x": (_c(100.0), _c(160.0))},
        )
        log_expr_result = dist.log_expression(context)
        expr_result = dist._expression(context)
        log_expr_val = float(pytensor.function([x_var], log_expr_result)([130.0])[0])
        expr_val = float(pytensor.function([x_var], expr_result)([130.0])[0])
        assert log_expr_val == pytest.approx(math.log(expr_val), rel=1e-10)
