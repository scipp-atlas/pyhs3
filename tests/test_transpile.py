"""
Tests for pyhs3.transpile — jaxify() + JaxifiedGraph.

Tests are gated with module-level pytest.importorskip("jax") so the suite stays
green in environments that only have pytensor (no JAX).
"""

from __future__ import annotations

import math
import sys

import pytensor.tensor as pt
import pytest
from scipy.stats import norm

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
                )[0]
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

    def test_jaxify_raises_on_unnamed_explicit_input(self):
        """Explicit inputs= with an unnamed variable raises ValueError."""
        unnamed = pt.scalar()  # no name
        with pytest.raises(ValueError, match="named"):
            jaxify(unnamed, inputs=[unnamed])

    def test_jaxify_raises_on_duplicate_names(self):
        """Two inputs sharing a name raises ValueError."""
        x1 = pt.scalar("x")
        x2 = pt.scalar("x")  # same name as x1
        expr = x1 + x2
        with pytest.raises(ValueError, match="unique"):
            jaxify(expr, inputs=[x1, x2])


# ---------------------------------------------------------------------------
# JaxifiedGraph.__call__ — kwargs validation
# ---------------------------------------------------------------------------


class TestJaxifiedGraphCall:
    def test_call_by_kwargs(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        val = jg(x=jnp.float64(0.0), mu=jnp.float64(0.0), sigma=jnp.float64(1.0))
        assert jnp.isfinite(val[0])

    def test_call_raises_on_extra_kwarg(self):
        """Extra kwargs beyond the graph inputs raise TypeError (Python's own error)."""
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            jg(
                x=jnp.float64(0.0),
                mu=jnp.float64(0.0),
                sigma=jnp.float64(1.0),
                extra=jnp.float64(0.0),
            )

    def test_call_raises_on_missing_kwarg(self):
        """Missing kwargs raise TypeError (Python's own error for missing args)."""
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        with pytest.raises(TypeError, match="missing"):
            jg(x=jnp.float64(0.0), mu=jnp.float64(0.0))  # sigma omitted

    def test_call_positional(self):
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        ordered = [jnp.float64(0.0)] * len(jg.input_names)
        args = list(ordered)
        sigma_idx = jg.input_names.index("sigma")
        args[sigma_idx] = jnp.float64(1.0)
        val = jg.call_positional(*args)
        assert jnp.isfinite(val[0])

    def test_jaxify_importerror_when_jax_dispatch_missing(self, monkeypatch):
        """jaxify() raises ImportError with a helpful message when pytensor[jax] absent."""
        monkeypatch.setitem(sys.modules, "pytensor.link.jax.dispatch.basic", None)
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        with pytest.raises(ImportError, match=r"pyhs3\.transpile requires JAX"):
            jaxify(pdf_expr)

    def test_pytree_dict_usage(self):
        """Typical everwillow/optimistix pattern: nll takes a dict pytree."""
        _, _, _, pdf_expr = _gaussian_pytensor_expr()
        jg = jaxify(pdf_expr)
        fixed_x = jnp.float64(0.0)

        @jax.jit
        def nll(free_params: dict) -> object:
            all_params = {**free_params, "x": fixed_x}
            return -2 * jnp.log(jg(**all_params)[0])

        free = {"mu": jnp.float64(0.0), "sigma": jnp.float64(1.0)}
        result = nll(free)
        assert jnp.isfinite(result)
