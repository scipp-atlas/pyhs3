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

import numpy as np
import pytensor
import pytest

import pyhs3 as hs3
from pyhs3.data import BinnedData, Data
from pyhs3.distributions import Distributions, HistFactoryDistChannel
from pyhs3.domains import Domains, ProductDomain
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoints, ParameterSet
from pyhs3.workspace import Workspace

# scipy.stats.poisson.logpmf is preferred for the analytic reference; fall back
# to gammaln if scipy is unavailable in the test environment.
try:
    from scipy.stats import poisson as _scipy_poisson

    def _poisson_logpmf(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
        return np.asarray(_scipy_poisson.logpmf(counts, rates), dtype=np.float64)

except ModuleNotFoundError:  # pragma: no cover - scipy is in the dev deps

    def _poisson_logpmf(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
        from scipy.special import gammaln  # noqa: PLC0415

        return counts * np.log(rates) - rates - gammaln(counts + 1)


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
