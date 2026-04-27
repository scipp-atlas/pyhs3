"""
Tests for pyhs3.transpile — jaxify() + JaxifiedGraph.

All tests that actually execute JAX code are gated with
pytest.importorskip("jax") so the suite stays green in environments
that only have pytensor (no JAX).
"""

from __future__ import annotations

import math

import pytensor.tensor as pt
import pytest
from scipy.stats import norm

import pyhs3.transpile as _transpile
from pyhs3.transpile import JaxifiedGraph, jaxify

jax = pytest.importorskip("jax")
jnp = jax.numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_pytensor_expr() -> tuple[
    pt.TensorVariable, pt.TensorVariable, pt.TensorVariable, pt.TensorVariable
]:
    """Return (x, mu, sigma, pdf_expr) as pytensor scalars."""
    x = pt.scalar("x")
    mu = pt.scalar("mu")
    sigma = pt.scalar("sigma")
    pdf_expr = pt.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
        sigma * pt.sqrt(pt.constant(2 * math.pi, dtype="float64"))
    )
    return x, mu, sigma, pdf_expr


# ---------------------------------------------------------------------------
# jaxify — basic smoke tests
# ---------------------------------------------------------------------------


class TestJaxifyGaussian:
    def test_jaxify_returns_jaxified_graph(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        result = jaxify(pdf_expr)
        assert isinstance(result, JaxifiedGraph)

    def test_jaxified_graph_has_correct_input_names(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        result = jaxify(pdf_expr)
        assert set(result.input_names) == {"x", "mu", "sigma"}

    def test_jaxify_matches_scipy_gaussian(self):
        """jaxify(pdf_expr)(**kwargs) matches scipy.stats.norm.pdf to 1e-10."""
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)

        mu_val = 0.5
        sigma_val = 1.2
        x_vals = [-1.0, 0.0, 0.5, 1.0, 2.0]
        for xv in x_vals:
            got = float(
                jg(
                    x=jnp.float64(xv),
                    mu=jnp.float64(mu_val),
                    sigma=jnp.float64(sigma_val),
                )
            )
            expected = norm.pdf(xv, loc=mu_val, scale=sigma_val)
            assert abs(got - expected) < 1e-10, (
                f"x={xv}: got {got}, expected {expected}"
            )

    def test_jaxify_explicit_inputs_subset(self):
        """When inputs= is provided, only those appear in the graph."""
        x, mu, _, _ = _gaussian_pytensor_expr()
        fixed_sigma = pt.constant(1.0, dtype="float64")
        pinned_expr = pt.exp(-0.5 * ((x - mu) / fixed_sigma) ** 2) / (
            fixed_sigma * pt.sqrt(pt.constant(2 * math.pi, dtype="float64"))
        )
        jg = jaxify(pinned_expr, inputs=[x, mu])
        assert set(jg.input_names) == {"x", "mu"}


# ---------------------------------------------------------------------------
# JaxifiedGraph.__call__ and call_positional
# ---------------------------------------------------------------------------


class TestJaxifiedGraphCall:
    def test_call_by_kwargs(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        val = jg(x=jnp.float64(0.0), mu=jnp.float64(0.0), sigma=jnp.float64(1.0))
        assert jnp.isfinite(val)

    def test_call_positional(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        ordered = [jnp.float64(0.0)] * len(jg.input_names)
        args = list(ordered)
        sigma_idx = jg.input_names.index("sigma")
        args[sigma_idx] = jnp.float64(1.0)
        val = jg.call_positional(*args)
        assert jnp.isfinite(val)


# ---------------------------------------------------------------------------
# JaxifiedGraph.with_partition
# ---------------------------------------------------------------------------


class TestWithPartition:
    def test_with_partition_returns_callable(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        f = jg.with_partition(free=["mu", "sigma"], fixed=["x"])
        assert callable(f)

    def test_with_partition_evaluates_correctly(self):
        """f(free_vec, fixed_vec) -> scalar matches direct call."""
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)

        free_names = ["mu", "sigma"]
        fixed_names = ["x"]
        f = jg.with_partition(free=free_names, fixed=fixed_names)

        free_vec = jnp.array([0.5, 1.2])  # mu=0.5, sigma=1.2
        fixed_vec = jnp.array([0.0])  # x=0.0
        got = float(f(free_vec, fixed_vec))
        expected = norm.pdf(0.0, loc=0.5, scale=1.2)
        assert abs(got - expected) < 1e-10

    def test_with_partition_raises_on_unknown_name(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        with pytest.raises((KeyError, ValueError)):
            jg.with_partition(free=["nonexistent"], fixed=["x", "mu", "sigma"])


# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


class TestImportErrorPath:
    def test_jaxify_raises_friendly_error_when_jax_missing(self, monkeypatch):
        """When _IMPORT_ERROR is set, jaxify raises ImportError with helpful message."""
        original = _transpile._IMPORT_ERROR
        try:
            monkeypatch.setattr(_transpile, "_IMPORT_ERROR", ImportError("no jax"))
            with pytest.raises(ImportError, match=r"pyhs3\.transpile requires JAX"):
                _transpile.jaxify(None)
        finally:
            monkeypatch.setattr(_transpile, "_IMPORT_ERROR", original)
