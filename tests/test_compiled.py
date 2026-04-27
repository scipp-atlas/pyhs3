"""
Tests for pyhs3.compiled — CompiledLikelihood / Analysis.compile().
"""

from __future__ import annotations

import copy

import pytensor
import pytest
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.stats import truncnorm

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from pyhs3 import CompiledLikelihood, Workspace
from pyhs3.transpile import JaxifiedGraph

# ---------------------------------------------------------------------------
# Minimal two-Gaussian-channel workspace fixture
# Shared parameter: "mean"; each channel has its own observable.
# Wide axis bounds (±10) make Gaussian truncation negligible vs full-support.
# ---------------------------------------------------------------------------

_WS_DICT: dict = {
    "metadata": {"hs3_version": "0.2"},
    "distributions": [
        {
            "name": "gauss1",
            "type": "gaussian_dist",
            "x": "x_obs",
            "mean": "mean",
            "sigma": 1.0,
        },
        {
            "name": "gauss2",
            "type": "gaussian_dist",
            "x": "y_obs",
            "mean": "mean",
            "sigma": 2.0,
        },
    ],
    "domains": [
        {
            "name": "main",
            "type": "product_domain",
            "axes": [{"name": "mean", "min": -10.0, "max": 10.0}],
        }
    ],
    "data": [
        {
            "name": "data1",
            "type": "unbinned",
            "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
            "entries": [[1.0], [2.0], [3.0], [4.0], [5.0]],
        },
        {
            "name": "data2",
            "type": "unbinned",
            "axes": [{"name": "y_obs", "min": -10.0, "max": 10.0}],
            "entries": [[0.5], [1.5], [2.5], [3.5], [4.5]],
        },
    ],
    "likelihoods": [
        {"name": "L", "distributions": ["gauss1", "gauss2"], "data": ["data1", "data2"]}
    ],
    "analyses": [
        {"name": "A", "likelihood": "L", "domains": ["main"], "init": "params"}
    ],
    "parameter_points": [
        {"name": "params", "parameters": [{"name": "mean", "value": 2.0}]}
    ],
}


def _ws() -> Workspace:
    return Workspace(**_WS_DICT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncnorm_logpdf(
    x: float, loc: float, scale: float, low: float, high: float
) -> float:
    a = (low - loc) / scale
    b = (high - loc) / scale
    return float(truncnorm.logpdf(x, a, b, loc=loc, scale=scale))


# ---------------------------------------------------------------------------
# Workspace backref
# ---------------------------------------------------------------------------


def test_analysis_workspace_backref():
    ws = _ws()
    analysis = ws.analyses["A"]
    assert analysis._workspace is ws


# ---------------------------------------------------------------------------
# CompiledLikelihood construction
# ---------------------------------------------------------------------------


def test_compile_returns_compiled_likelihood():
    compiled = _ws().analyses["A"].compile()
    assert isinstance(compiled, CompiledLikelihood)


def test_compile_exposes_nll_expr():
    compiled = _ws().analyses["A"].compile()
    # nll_expr is a PyTensor TensorVariable (has .type attribute)
    assert hasattr(compiled.nll_expr, "type")


def test_compile_classifies_free_parameters():
    compiled = _ws().analyses["A"].compile()
    assert "mean" in compiled.free_parameters
    assert "mean" not in compiled.fixed_parameters


def test_compile_const_parameter_is_fixed():
    ws_dict = copy.deepcopy(_WS_DICT)
    ws_dict["parameter_points"][0]["parameters"][0]["const"] = True
    ws = Workspace(**ws_dict)
    compiled = ws.analyses["A"].compile()
    assert "mean" in compiled.fixed_parameters
    assert "mean" not in compiled.free_parameters


# ---------------------------------------------------------------------------
# Numerical accuracy — two Gaussian channels
# ---------------------------------------------------------------------------


def test_compile_nll_matches_truncnorm():
    """CompiledLikelihood NLL matches scipy truncnorm log-prob to within 1e-6."""
    compiled = _ws().analyses["A"].compile()

    inputs = {
        v.name: v
        for v in explicit_graph_inputs([compiled.nll_expr])
        if v.name is not None
    }
    fn = pytensor.function([inputs["mean"]], compiled.nll_expr)
    nll_val = float(fn(2.0))

    events1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    events2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    lp = sum(_truncnorm_logpdf(x, 2.0, 1.0, -10.0, 10.0) for x in events1) + sum(
        _truncnorm_logpdf(y, 2.0, 2.0, -10.0, 10.0) for y in events2
    )
    expected = -2.0 * lp

    assert abs(nll_val - expected) < 1e-6, f"got {nll_val}, expected {expected}"


# ---------------------------------------------------------------------------
# JAX integration (skipped when JAX unavailable)
# ---------------------------------------------------------------------------

_skip_no_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


@_skip_no_jax
def test_compile_to_jax_returns_jaxified_graph():
    compiled = _ws().analyses["A"].compile()
    jg = compiled.to_jax()
    assert isinstance(jg, JaxifiedGraph)
    assert "mean" in jg.input_names


@_skip_no_jax
def test_compile_to_jax_evaluates():
    compiled = _ws().analyses["A"].compile()
    jg = compiled.to_jax()
    val = jg(mean=jnp.float64(2.0))[0]
    assert jnp.isfinite(val)
