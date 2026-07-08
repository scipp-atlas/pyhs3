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
from typing import Literal, cast

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
    LandauDist,
    LogNormalDist,
    PoissonDist,
    UniformDist,
)
from pyhs3.distributions.cms import (
    FastVerticalInterpHistPdf2D2Dist,
    FastVerticalInterpHistPdf2Dist,
    GGZZBackgroundDist,
    QQZZBackgroundDist,
)
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.mathematical import (
    BernsteinPolyDist,
    GenericDist,
    PolynomialDist,
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
    from scipy.stats import expon as _scipy_expon
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

    def _expon_logpdf(x: np.ndarray, c: np.ndarray) -> np.ndarray:
        # ExponentialDist's rate parametrization (pdf = c * exp(-c*x)) maps onto
        # scipy's loc/scale form as scale = 1/c.
        return np.asarray(_scipy_expon.logpdf(x, scale=1.0 / c), dtype=np.float64)

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

    def _expon_logpdf(x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.log(c) - c * x


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


class _NoLogOverrideDist(Distribution):
    """Minimal Distribution subclass with no ``log_likelihood`` override.

    As of Phase 3 of #254, every basic.py distribution has an analytic
    ``log_likelihood`` override, so none of them exercise the base class's
    ``pt.log(likelihood())`` default directly anymore. This standalone class
    keeps that fallback path covered.
    """

    type: Literal["_no_log_override_dist"] = "_no_log_override_dist"
    x: str | float | int
    c: str | float | int

    def likelihood(self, context: Context) -> TensorVar:
        x = context[self._parameters["x"]]
        c = context[self._parameters["c"]]
        return cast(TensorVar, c * pt.exp(-c * x))


class TestBaseDistributionLogLikelihoodDefault:
    """A distribution with no analytic override falls back to pt.log(likelihood())."""

    @pytest.mark.parametrize(
        ("x", "c"),
        [
            pytest.param(1.0, 0.5, id="ordinary_point"),
            pytest.param(3.0, 2.0, id="larger_rate"),
        ],
    )
    def test_default_log_likelihood_matches_log_of_likelihood(self, x, c):
        """With no log_likelihood override, log_likelihood() falls back to log(likelihood())."""
        dist = _NoLogOverrideDist(name="e", x="x", c="c")
        context = Context({"x": _c(x), "c": _c(c)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        assert log_val == pytest.approx(np.log(prob_val), rel=1e-12)


class TestExponentialLogLikelihood:
    """ExponentialDist.log_likelihood is the analytic log form of likelihood()."""

    @pytest.mark.parametrize(
        ("x", "c"),
        [
            pytest.param(1.0, 0.5, id="ordinary_point"),
            pytest.param(3.0, 2.0, id="larger_rate"),
            pytest.param(0.0, 1.0, id="at_origin"),
        ],
    )
    def test_matches_log_of_likelihood(self, x, c):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = ExponentialDist(name="e", x="x", c="c")
        context = Context({"x": _c(x), "c": _c(c)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    @pytest.mark.parametrize(
        ("x", "c"),
        [
            pytest.param(1.0, 0.5, id="ordinary_point"),
            pytest.param(3.0, 2.0, id="larger_rate"),
        ],
    )
    def test_matches_scipy_expon_logpdf(self, x, c):
        """likelihood() is scipy's expon pdf with scale=1/c, matching exactly."""
        dist = ExponentialDist(name="e", x="x", c="c")
        context = Context({"x": _c(x), "c": _c(c)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        expected = float(_expon_logpdf(np.array(x), np.array(c)))
        assert log_val == pytest.approx(expected, rel=1e-10)

    def test_finite_deep_in_tail_while_likelihood_underflows(self):
        """c*x = 800: likelihood underflows to 0.0, log_likelihood stays finite (~-800)."""
        dist = ExponentialDist(name="e", x="x", c="c")
        context = Context({"x": _c(800.0), "c": _c(1.0)})
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)
        assert log_val == pytest.approx(-800.0, rel=1e-10)


class TestUniformLogLikelihood:
    """UniformDist.log_likelihood is the analytic log form of likelihood()."""

    def test_matches_log_of_likelihood(self):
        """log_likelihood equals log(likelihood): both are parameter-independent constants."""
        dist = UniformDist(name="u", x=["x"])
        context = Context({"x": _c(1.23)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)
        assert log_val == pytest.approx(0.0, abs=1e-12)


class TestLandauLogLikelihood:
    """LandauDist.log_likelihood is the analytic log form of likelihood().

    The raw density is a product of two exponentials divided by a constant
    normalization, so its log is a plain sum -- no product/sum term can
    partially cancel another, unlike e.g. a piecewise density built from a
    difference of two comparable-magnitude terms.
    """

    @pytest.mark.parametrize(
        ("x", "mean", "sigma"),
        [
            pytest.param(0.0, 0.0, 1.0, id="at_mode"),
            pytest.param(2.0, 0.0, 1.0, id="right_tail"),
            pytest.param(-1.5, 0.0, 1.0, id="left_of_mode"),
            pytest.param(130.0, 125.0, 10.0, id="hep_scale"),
        ],
    )
    def test_matches_log_of_likelihood(self, x, mean, sigma):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = LandauDist(name="landau", x="x", mean="mean", sigma="sigma")
        context = Context({"x": _c(x), "mean": _c(mean), "sigma": _c(sigma)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-10)

    def test_finite_deep_in_tail_while_likelihood_underflows(self):
        """z = (x-mean)/sigma = 100 deep in the tail: likelihood underflows, log_likelihood stays finite."""
        dist = LandauDist(name="landau", x="x", mean="mean", sigma="sigma")
        context = Context({"x": _c(100.0), "mean": _c(0.0), "sigma": _c(1.0)})
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)


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


# --------------------------------------------------------------------------
# Layer 2 of #243: modifier.log_constraint() and the HistFactoryDistChannel /
# Model log-space constraint assembly built on top of it, so neither
# HistFactoryDistChannel.log_extended_likelihood nor Model.log_prob ever
# takes pt.log() of a probability-space constraint that can underflow to 0.0.
# --------------------------------------------------------------------------


def _normsys_workspace(alpha_value: float) -> Workspace:
    """Single-channel, single-bin HFDC workspace with one normsys (Gaussian)
    constraint on ``alpha_lumi``, observed data equal to the nominal yield.
    """
    obs_name = "x_SR"
    channel = HistFactoryDistChannel(
        name="SR",
        axes=[{"name": obs_name, "min": 0.0, "max": 10.0, "nbins": 1}],
        samples=[
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [
                    {
                        "name": "lumi",
                        "type": "normsys",
                        "parameter": "alpha_lumi",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9},
                    }
                ],
            }
        ],
    )
    binned = BinnedData(
        name="SR_data",
        axes=[{"name": obs_name, "min": 0.0, "max": 10.0, "nbins": 1}],
        contents=[10.0],
    )
    return Workspace(
        metadata=Metadata(hs3_version="0.3.0"),
        distributions=Distributions([channel]),
        data=Data([binned]),
        likelihoods=Likelihoods(
            [Likelihood(name="L", distributions=[channel], data=[binned])]
        ),
        domains=Domains([ProductDomain(name="default")]),
        parameter_points=ParameterPoints(
            [
                ParameterSet(
                    name="default",
                    parameters=[{"name": "alpha_lumi", "value": alpha_value}],
                )
            ]
        ),
    )


def test_log_prob_finite_for_normsys_constraint_at_alpha_40():
    """model.log_prob stays finite when the normsys nuisance is pushed to
    |alpha|=40, where the probability-space Gaussian constraint underflows to
    0.0 (exp(-40**2/2) < 2.2e-308).  See issue #243.
    """
    ws = _normsys_workspace(alpha_value=40.0)
    model = ws.model(next(iter(ws.likelihoods)), progress=False)

    # The probability-space constraint underflows to 0.0 at |alpha|=40.
    dist = next(iter(ws.distributions))
    _, modifier, sample_data = next(dist.constraint_specs())
    prob_constraint = float(
        np.asarray(
            modifier.make_constraint(
                Context({"alpha_lumi": pt.constant(40.0, dtype="float64")}),
                sample_data,
            ).eval()
        )
    )
    assert prob_constraint == 0.0

    lp = model.log_prob
    inputs = {
        v.name: v
        for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
        if v.name
    }
    fn = pytensor.function(list(inputs.values()), lp)
    val = float(np.asarray(fn(**model.data, **model.nominal_params)).item())
    assert math.isfinite(val)

    # code4 interpolation for alpha >= alpha0=1: factor = hi**alpha (nominal
    # factor is 1.0), so expected_rate = contents * hi**alpha.
    expected_rate = 10.0 * (1.1**40.0)
    expected = float(
        _poisson_logpmf(np.array([10.0]), np.array([expected_rate])).sum()
        + _norm_logpdf(np.array([40.0]), np.array([0.0]), np.array([1.0])).sum()
    )
    assert val == pytest.approx(expected, rel=1e-8)


def test_logpdf_finite_for_normsys_constraint_at_alpha_40():
    """logpdf() for a single HFDC channel with a normsys constraint at
    |alpha|=40 stays finite and matches the analytic sum of the Poisson
    log-pmf term and scipy.stats.norm.logpdf(alpha) for the constraint. See
    issue #243.
    """
    ws = _normsys_workspace(alpha_value=40.0)
    model = ws.model(next(iter(ws.likelihoods)), progress=False)

    logpdf_val = float(
        np.asarray(
            model.logpdf_unsafe("SR", alpha_lumi=40.0, SR_observed=np.array([10.0]))
        )
    )
    assert math.isfinite(logpdf_val)

    expected_rate = 10.0 * (1.1**40.0)
    expected = float(
        _poisson_logpmf(np.array([10.0]), np.array([expected_rate])).sum()
        + _norm_logpdf(np.array([40.0]), np.array([0.0]), np.array([1.0])).sum()
    )
    assert logpdf_val == pytest.approx(expected, rel=1e-8)


# --------------------------------------------------------------------------
# Phase 3 of #254: GenericDist, PolynomialDist, and BernsteinPolyDist are
# explicit drop-downs — they implement only likelihood() and inherit the base
# class's log_likelihood = pt.log(likelihood). These tests characterize that
# behavior (including the NaN consequence for signed polynomials) rather than
# asserting an analytic log form, since none exists for these classes.
# --------------------------------------------------------------------------


class TestGenericDistLogLikelihood:
    """GenericDist has no log_likelihood override: it uses the base default.

    The expression is an arbitrary user-supplied string with no general
    analytic log form, so the drop-down to pt.log(likelihood) is permanent
    rather than a placeholder awaiting a Phase 3 conversion.
    """

    @pytest.mark.parametrize(
        ("expression", "params"),
        [
            pytest.param("x**2 + 1", {"x": 2.0}, id="polynomial_positive"),
            pytest.param(
                "exp(-x**2/2)", {"x": 1.5}, id="gaussian_kernel_always_positive"
            ),
            pytest.param("sin(x) + 2", {"x": 0.7}, id="shifted_sine_always_positive"),
        ],
    )
    def test_matches_log_of_likelihood(self, expression, params):
        """log_likelihood equals log(likelihood) wherever the expression is positive.

        sympy_to_pytensor keys its variable substitution off each PyTensor
        variable's own ``.name`` (not the context dict key), so the constants
        here must be named to match the free symbols in the expression.
        """
        dist = GenericDist(name="g", expression=expression)
        context = Context(
            {
                name: cast(TensorVar, pt.constant(value, dtype="float64", name=name))
                for name, value in params.items()
            }
        )
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)


class TestPolynomialDistLogLikelihood:
    """PolynomialDist has no log_likelihood override: it uses the base default.

    Unlike Gaussian/Poisson/LogNormal, the raw polynomial value has no
    positivity constraint on its coefficients, so there is no analytic log
    form to author — the drop-down documented on the class is permanent.
    """

    @pytest.mark.parametrize(
        ("x", "coeffs"),
        [
            pytest.param(0.0, (1.0, 2.0, 3.0), id="quadratic_at_zero"),
            pytest.param(1.0, (1.0, 2.0, 3.0), id="quadratic_at_one"),
            pytest.param(2.0, (5.0, 3.0), id="linear_positive_slope"),
        ],
    )
    def test_matches_log_of_likelihood(self, x, coeffs):
        """log_likelihood equals log(likelihood) where the polynomial is positive."""
        coeff_names = [f"a{i}" for i in range(len(coeffs))]
        dist = PolynomialDist(name="p", x="x", coefficients=coeff_names)
        context = Context(
            {"x": _c(x)}
            | {name: _c(value) for name, value in zip(coeff_names, coeffs, strict=True)}
        )
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    def test_negative_polynomial_gives_nan_log_likelihood(self):
        """Documented drop-down consequence: a signed polynomial can go negative,
        and pt.log() of a negative value is NaN by construction (not a bug to
        fix — PolynomialDist places no positivity constraint on coefficients).
        """
        # 1 - 3x is negative for x > 1/3.
        dist = PolynomialDist(name="p", x="x", coefficients=["a0", "a1"])
        context = Context({"x": _c(1.0), "a0": _c(1.0), "a1": _c(-3.0)})
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val < 0.0
        assert math.isnan(log_val)


class TestBernsteinPolyDistLogLikelihood:
    """BernsteinPolyDist has no log_likelihood override: it uses the base default.

    The Bernstein basis polynomials are individually non-negative on [0, 1],
    but the coefficients are unconstrained, so the weighted sum can still go
    negative — there is no analytic log form to author here either.
    """

    @pytest.mark.parametrize(
        ("x", "coeffs"),
        [
            pytest.param(0.5, (2.0,), id="degree_zero_constant"),
            pytest.param(0.3, (1.0, 3.0), id="degree_one_linear"),
            pytest.param(0.6, (1.0, 4.0, 2.0), id="degree_two_quadratic"),
        ],
    )
    def test_matches_log_of_likelihood(self, x, coeffs):
        """log_likelihood equals log(likelihood) where the basis sum is positive."""
        coeff_names = [f"c{i}" for i in range(len(coeffs))]
        dist = BernsteinPolyDist(name="b", x="x", coefficients=coeff_names)
        context = Context(
            {"x": _c(x)}
            | {name: _c(value) for name, value in zip(coeff_names, coeffs, strict=True)}
        )
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    def test_negative_bernstein_sum_gives_nan_log_likelihood(self):
        """Documented drop-down consequence: unconstrained coefficients can make
        the weighted basis sum negative, and pt.log() of a negative value is
        NaN by construction (not a bug to fix).
        """
        # B_0,1(0.9) = 0.1, B_1,1(0.9) = 0.9 -> -5*0.1 + 1*0.9 = 0.4 > 0 doesn't
        # go negative; use a stronger negative weight on the dominant basis term.
        dist = BernsteinPolyDist(name="b", x="x", coefficients=["c0", "c1"])
        context = Context({"x": _c(0.9), "c0": _c(1.0), "c1": _c(-5.0)})
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val < 0.0
        assert math.isnan(log_val)


# --------------------------------------------------------------------------
# Phase 3.5 of #254: analytic log_likelihood() on the CMS ggZZ/qqZZ background
# distributions (products of a power-law and an exponential decay, so the log
# is a clean sum). The CMS FastVerticalInterp placeholders stay on the base
# pt.log(likelihood()) fallback (see class docstrings for why).
# --------------------------------------------------------------------------


class TestGGZZBackgroundLogLikelihood:
    """GGZZBackgroundDist.log_likelihood is the analytic log form of likelihood().

    likelihood() = a1 * m4l**a2 * exp(-a3 * m4l), so the analytic log is
    log(a1) + a2 * log(m4l) - a3 * m4l.
    """

    @pytest.mark.parametrize(
        ("a1", "a2", "a3", "m4l"),
        [
            pytest.param(1.0, 2.0, 0.05, 200.0, id="typical"),
            pytest.param(0.5, 1.5, 0.02, 150.0, id="smaller_norm"),
            pytest.param(2.0, -1.0, 0.03, 300.0, id="negative_power"),
        ],
    )
    def test_matches_log_of_likelihood(self, a1, a2, a3, m4l):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = GGZZBackgroundDist(name="ggzz", m4l="m4l", a1="a1", a2="a2", a3="a3")
        context = Context({"m4l": _c(m4l), "a1": _c(a1), "a2": _c(a2), "a3": _c(a3)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    def test_finite_at_large_mass_while_likelihood_underflows(self):
        """a3 * m4l = 800: exp(-800) underflows likelihood to 0.0, log stays finite."""
        a1, a2, a3, m4l = 1.0, 2.0, 1.0, 800.0
        dist = GGZZBackgroundDist(name="ggzz", m4l="m4l", a1="a1", a2="a2", a3="a3")
        context = Context({"m4l": _c(m4l), "a1": _c(a1), "a2": _c(a2), "a3": _c(a3)})
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)
        expected = math.log(a1) + a2 * math.log(m4l) - a3 * m4l
        assert log_val == pytest.approx(expected, rel=1e-10)


class TestQQZZBackgroundLogLikelihood:
    """QQZZBackgroundDist.log_likelihood is the analytic log form of likelihood().

    likelihood() = a1 * (m4l + a2)**a3 * exp(-a4 * m4l), so the analytic log is
    log(a1) + a3 * log(m4l + a2) - a4 * m4l.
    """

    @pytest.mark.parametrize(
        ("a1", "a2", "a3", "a4", "m4l"),
        [
            pytest.param(1.0, 10.0, 2.0, 0.05, 200.0, id="typical"),
            pytest.param(0.3, 5.0, 1.2, 0.02, 150.0, id="smaller_norm"),
            pytest.param(2.0, -50.0, 1.5, 0.03, 300.0, id="negative_shift"),
        ],
    )
    def test_matches_log_of_likelihood(self, a1, a2, a3, a4, m4l):
        """log_likelihood equals log(likelihood) at ordinary parameter points."""
        dist = QQZZBackgroundDist(
            name="qqzz", m4l="m4l", a1="a1", a2="a2", a3="a3", a4="a4"
        )
        context = Context(
            {
                "m4l": _c(m4l),
                "a1": _c(a1),
                "a2": _c(a2),
                "a3": _c(a3),
                "a4": _c(a4),
            }
        )
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        np.testing.assert_allclose(log_val, np.log(prob_val), rtol=1e-12)

    def test_finite_at_large_mass_while_likelihood_underflows(self):
        """a4 * m4l = 800: exp(-800) underflows likelihood to 0.0, log stays finite."""
        a1, a2, a3, a4, m4l = 1.0, 10.0, 2.0, 1.0, 800.0
        dist = QQZZBackgroundDist(
            name="qqzz", m4l="m4l", a1="a1", a2="a2", a3="a3", a4="a4"
        )
        context = Context(
            {
                "m4l": _c(m4l),
                "a1": _c(a1),
                "a2": _c(a2),
                "a3": _c(a3),
                "a4": _c(a4),
            }
        )
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        log_val = float(pytensor.function([], dist.log_likelihood(context))())

        assert prob_val == 0.0
        assert math.isfinite(log_val)
        expected = math.log(a1) + a3 * math.log(m4l + a2) - a4 * m4l
        assert log_val == pytest.approx(expected, rel=1e-10)


class TestFastVerticalInterpPlaceholdersUseBaseDefault:
    """The FastVerticalInterp morphs are simplified placeholders (per their
    docstrings) with no analytic log form; both stay on the base
    pt.log(likelihood()) fallback until a real morphing implementation lands.
    """

    def test_fastverticalinterphistpdf2_matches_base_default(self):
        dist = FastVerticalInterpHistPdf2Dist(
            name="fvh2", x="x", coefList=["coef0", "coef1"]
        )
        context = Context({"x": _c(1.0), "coef0": _c(0.2), "coef1": _c(-0.3)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        assert log_val == pytest.approx(np.log(prob_val), rel=1e-12)

    def test_fastverticalinterphistpdf2d2_matches_base_default(self):
        dist = FastVerticalInterpHistPdf2D2Dist(
            name="fvh2d2", x="x", y="y", coefList=["coef0"]
        )
        context = Context({"x": _c(1.0), "y": _c(2.0), "coef0": _c(0.4)})
        log_val = float(pytensor.function([], dist.log_likelihood(context))())
        prob_val = float(pytensor.function([], dist.likelihood(context))())
        assert log_val == pytest.approx(np.log(prob_val), rel=1e-12)
