"""
Tests for ws.model(analysis) / ws.model(likelihood) — joint log-prob interface.
"""

from __future__ import annotations

import numpy as np
import pytensor
import pytest
from scipy.stats import truncnorm

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor import TensorConstant

from pyhs3 import Workspace, jaxify
from pyhs3.distributions import Distributions
from pyhs3.domains import ProductDomain
from pyhs3.functions import Functions
from pyhs3.likelihoods import Likelihood
from pyhs3.model import Model
from pyhs3.parameter_points import ParameterSet

# ---------------------------------------------------------------------------
# Minimal two-Gaussian-channel workspace
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


def _truncnorm_logpdf(
    x: float, loc: float, scale: float, low: float, high: float
) -> float:
    a = (low - loc) / scale
    b = (high - loc) / scale
    return float(truncnorm.logpdf(x, a, b, loc=loc, scale=scale))


# ---------------------------------------------------------------------------
# ws.model(analysis) construction
# ---------------------------------------------------------------------------


def test_model_from_analysis_returns_model():
    ws = _ws()
    model = ws.model(ws.analyses["A"])
    assert isinstance(model, Model)


def test_model_from_analysis_auto_derives_domain():
    ws = _ws()
    model = ws.model(ws.analyses["A"])
    # mean should be bounded to [-10, 10] (from analysis domain)
    assert "mean" in model.parameters


def test_model_from_likelihood_returns_model():
    ws = _ws()
    model = ws.model(ws.likelihoods["L"])
    assert isinstance(model, Model)


def test_model_legacy_int_still_works():
    ws = _ws()
    model = ws.model(0)
    assert isinstance(model, Model)


def test_model_from_analysis_no_init_uses_empty_parameterset():
    """ws.model(analysis) with no init uses an empty default ParameterSet."""
    ws = Workspace(
        **{
            **_WS_DICT,
            "analyses": [
                # no "init" key → analysis.init is None → param_set = None branch
                {"name": "A", "likelihood": "L", "domains": ["main"]}
            ],
        }
    )
    model = ws.model(ws.analyses["A"], progress=False)
    assert isinstance(model, Model)
    assert model.parameterset.name == "default"


def test_model_from_analysis_no_parameter_points_raises():
    """ws.model(analysis) raises when workspace has no parameter_points but analysis has init."""
    ws = Workspace(
        **{
            **_WS_DICT,
            # Explicitly pass parameter_points=None so the workspace has none
            "parameter_points": None,
            "analyses": [
                {"name": "A", "likelihood": "L", "domains": ["main"], "init": "params"}
            ],
        }
    )
    with pytest.raises(ValueError, match="no parameter_points"):
        ws.model(ws.analyses["A"], progress=False)


def test_model_from_analysis_unknown_init_raises():
    """ws.model(analysis) raises when analysis.init references a non-existent parameter set."""
    ws_bad = Workspace(
        **{
            **_WS_DICT,
            "analyses": [
                {
                    "name": "A",
                    "likelihood": "L",
                    "domains": ["main"],
                    "init": "nonexistent_params",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="nonexistent_params"):
        ws_bad.model(ws_bad.analyses["A"], progress=False)


def test_model_from_analysis_multi_domain_merges():
    """ws.model(analysis) merges multiple domains into a single ProductDomain."""
    ws_multi = Workspace(
        **{
            **_WS_DICT,
            "domains": [
                {
                    "name": "main",
                    "type": "product_domain",
                    "axes": [{"name": "mean", "min": -10.0, "max": 10.0}],
                },
                {
                    "name": "nuis",
                    "type": "product_domain",
                    "axes": [{"name": "sigma_nuis", "min": -5.0, "max": 5.0}],
                },
            ],
            "analyses": [
                {
                    "name": "A",
                    "likelihood": "L",
                    "domains": ["main", "nuis"],
                    "init": "params",
                }
            ],
        }
    )
    model = ws_multi.model(ws_multi.analyses["A"], progress=False)
    assert "mean" in model.domain
    assert "sigma_nuis" in model.domain


# ---------------------------------------------------------------------------
# data and nominal_params properties
# ---------------------------------------------------------------------------


def test_likelihood_data_arrays_returns_numpy_dict():
    ws = _ws()
    d = ws.likelihoods["L"].data_arrays()
    assert "x_obs" in d
    assert "y_obs" in d
    np.testing.assert_array_equal(d["x_obs"], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(d["y_obs"], [0.5, 1.5, 2.5, 3.5, 4.5])


def test_likelihood_data_arrays_skips_binned_data():
    """BinnedData has no entries field → skipped by data_arrays()."""
    ws = Workspace(**_WS_BINNED)
    d = ws.likelihoods["L"].data_arrays()
    assert d == {}


def test_likelihood_data_arrays_skips_point_data():
    """PointData has axes=None by default → skipped by data_arrays()."""
    ws = Workspace(
        **{
            **_WS_DICT,
            "data": [
                {"name": "data1", "type": "point", "value": 1.0},
                _WS_DICT["data"][1],
            ],
        }
    )
    d = ws.likelihoods["L"].data_arrays()
    # data1 is PointData with axes=None → skipped; data2 (UnbinnedData) is included
    assert "x_obs" not in d
    assert "y_obs" in d


def test_log_prob_warns_for_weighted_data():
    ws_w = Workspace(
        **{
            **_WS_DICT,
            "data": [
                {
                    "name": "data1",
                    "type": "unbinned",
                    "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
                    "entries": [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    "weights": [1.0, 1.0, 1.0, 1.0, 0.0],
                },
                {
                    "name": "data2",
                    "type": "unbinned",
                    "axes": [{"name": "y_obs", "min": -10.0, "max": 10.0}],
                    "entries": [[0.5], [1.5], [2.5], [3.5], [4.5]],
                },
            ],
        }
    )
    model = ws_w.model(ws_w.analyses["A"], progress=False)
    with pytest.warns(UserWarning, match="weights"):
        _ = model.log_prob


def test_model_data_from_analysis():
    ws = _ws()
    model = ws.model(ws.analyses["A"])
    d = model.data
    assert "x_obs" in d
    assert "y_obs" in d
    np.testing.assert_array_equal(d["x_obs"], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(d["y_obs"], [0.5, 1.5, 2.5, 3.5, 4.5])


def test_model_nominal_params_from_analysis():
    ws = _ws()
    model = ws.model(ws.analyses["A"])
    p = model.nominal_params
    assert "mean" in p
    assert p["mean"] == pytest.approx(2.0)


def test_model_data_raises_without_likelihood():
    ws = _ws()
    model = ws.model(0)  # legacy, no likelihood
    with pytest.raises(RuntimeError, match="likelihood context"):
        _ = model.data


# ---------------------------------------------------------------------------
# log_prob property — structure
# ---------------------------------------------------------------------------


def test_log_prob_is_tensor_variable():
    ws = _ws()
    model = ws.model(ws.analyses["A"])
    assert hasattr(model.log_prob, "type")


def test_log_prob_raises_without_likelihood():
    ws = _ws()
    model = ws.model(0)
    with pytest.raises(RuntimeError, match="likelihood context"):
        _ = model.log_prob


def test_log_prob_has_free_symbolic_inputs():

    ws = _ws()
    model = ws.model(ws.analyses["A"])
    inputs = {v.name for v in explicit_graph_inputs([model.log_prob]) if v.name}
    # Both observables and parameters should be free symbolic inputs
    assert "mean" in inputs
    assert "x_obs" in inputs
    assert "y_obs" in inputs


# ---------------------------------------------------------------------------
# log_prob numerical accuracy
# ---------------------------------------------------------------------------


def test_log_prob_matches_truncnorm():
    """Joint log-prob evaluated at workspace defaults matches scipy truncnorm."""

    ws = _ws()
    model = ws.model(ws.analyses["A"])

    inputs_map = {
        v.name: v for v in explicit_graph_inputs([model.log_prob]) if v.name is not None
    }
    fn = pytensor.function(list(inputs_map.values()), model.log_prob)
    val = float(fn(**model.data, **model.nominal_params))

    events1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    events2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    lp = sum(_truncnorm_logpdf(x, 2.0, 1.0, -10.0, 10.0) for x in events1) + sum(
        _truncnorm_logpdf(y, 2.0, 2.0, -10.0, 10.0) for y in events2
    )

    assert abs(val - lp) < 1e-6, f"got {val}, expected {lp}"


def test_nll_is_minus_two_log_prob():
    """NLL = -2 * log_prob is correct."""

    ws = _ws()
    model = ws.model(ws.analyses["A"])
    nll_expr = -2.0 * model.log_prob

    inputs_map = {
        v.name: v for v in explicit_graph_inputs([nll_expr]) if v.name is not None
    }
    fn = pytensor.function(list(inputs_map.values()), nll_expr)
    nll_val = float(fn(**model.data, **model.nominal_params))

    events1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    events2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    lp = sum(_truncnorm_logpdf(x, 2.0, 1.0, -10.0, 10.0) for x in events1) + sum(
        _truncnorm_logpdf(y, 2.0, 2.0, -10.0, 10.0) for y in events2
    )
    expected = -2.0 * lp

    assert abs(nll_val - expected) < 1e-6, f"got {nll_val}, expected {expected}"


def test_log_prob_reusable_with_different_data():
    """Same Model evaluates correctly with different event data (same axis bounds)."""

    ws = _ws()
    model = ws.model(ws.analyses["A"])

    inputs_map = {
        v.name: v for v in explicit_graph_inputs([model.log_prob]) if v.name is not None
    }
    fn = pytensor.function(list(inputs_map.values()), model.log_prob)

    alt_data = {
        "x_obs": np.array([-1.0, 0.0, 1.0]),
        "y_obs": np.array([-0.5, 0.5, 1.5]),
    }
    val = float(fn(**alt_data, **model.nominal_params))

    expected_lp = sum(
        _truncnorm_logpdf(x, 2.0, 1.0, -10.0, 10.0) for x in alt_data["x_obs"]
    ) + sum(_truncnorm_logpdf(y, 2.0, 2.0, -10.0, 10.0) for y in alt_data["y_obs"])
    assert abs(val - expected_lp) < 1e-6


# ---------------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------------

_skip_no_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


@_skip_no_jax
def test_log_prob_jaxify_evaluates():

    ws = _ws()
    model = ws.model(ws.analyses["A"])
    jg = jaxify(model.log_prob)
    val = jg(**{k: jnp.array(v) for k, v in model.data.items()}, mean=jnp.float64(2.0))
    assert jnp.isfinite(val[0])


@_skip_no_jax
def test_nll_jaxify_matches_truncnorm():

    ws = _ws()
    model = ws.model(ws.analyses["A"])
    nll = -2.0 * model.log_prob
    jg = jaxify(nll)

    val = float(
        jg(**{k: jnp.array(v) for k, v in model.data.items()}, mean=jnp.float64(2.0))[0]
    )

    events1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    events2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    lp = sum(_truncnorm_logpdf(x, 2.0, 1.0, -10.0, 10.0) for x in events1) + sum(
        _truncnorm_logpdf(y, 2.0, 2.0, -10.0, 10.0) for y in events2
    )
    expected = -2.0 * lp

    assert abs(val - expected) < 1e-5, f"got {val}, expected {expected}"


# ---------------------------------------------------------------------------
# log_prob coverage: skipped / constant-fallback / aux_distributions paths
# ---------------------------------------------------------------------------

_WS_BINNED: dict = {
    "metadata": {"hs3_version": "0.2"},
    "distributions": [
        {
            "name": "signal",
            "type": "gaussian_dist",
            "x": "mass",
            "mean": "mu",
            "sigma": 1.0,
        }
    ],
    "domains": [
        {
            "name": "d",
            "type": "product_domain",
            "axes": [{"name": "mu", "min": -10.0, "max": 10.0}],
        }
    ],
    "data": [
        {
            "name": "binned_obs",
            "type": "binned",
            "contents": [10, 20, 15],
            "axes": [{"name": "mass", "edges": [110.0, 120.0, 130.0, 140.0]}],
        }
    ],
    "likelihoods": [{"name": "L", "distributions": ["signal"], "data": ["binned_obs"]}],
    "parameter_points": [{"name": "p", "parameters": [{"name": "mu", "value": 125.0}]}],
}

_WS_AUX: dict = {
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
            "name": "constraint",
            "type": "gaussian_dist",
            "x": "alpha",
            "mean": 0.0,
            "sigma": 1.0,
        },
    ],
    "domains": [
        {
            "name": "main",
            "type": "product_domain",
            "axes": [
                {"name": "mean", "min": -10.0, "max": 10.0},
                {"name": "alpha", "min": -5.0, "max": 5.0},
            ],
        }
    ],
    "data": [
        {
            "name": "data1",
            "type": "unbinned",
            "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
            "entries": [[1.0], [2.0], [3.0]],
        }
    ],
    "likelihoods": [
        {
            "name": "L",
            "distributions": ["gauss1"],
            "data": ["data1"],
            "aux_distributions": ["constraint"],
        }
    ],
    "analyses": [
        {"name": "A", "likelihood": "L", "domains": ["main"], "init": "params"}
    ],
    "parameter_points": [
        {
            "name": "params",
            "parameters": [
                {"name": "mean", "value": 0.0},
                {"name": "alpha", "value": 0.0},
            ],
        }
    ],
}


def test_log_prob_binned_data_has_no_entries_returns_constant_zero():
    """BinnedData has no entries; lp_terms stays empty → constant 0.0 fallback."""
    ws = Workspace(**_WS_BINNED)
    model = ws.model(ws.likelihoods["L"], progress=False)
    lp = model.log_prob
    # All data entries were BinnedData (no entries attribute) → skipped.
    # lp_terms is empty → pt.constant(0.0) is returned.
    assert isinstance(lp, TensorConstant)
    fn = pytensor.function([], lp)
    assert float(fn()) == pytest.approx(0.0)


def test_log_prob_string_datum_skipped_returns_constant_zero():
    """Likelihood with unresolved string datum refs skips all pairs → constant 0.0."""
    likelihood = Likelihood(name="L", distributions=["gauss1"], data=["data1"])
    model = Model(
        parameterset=ParameterSet(name="default", parameters=[]),
        distributions=Distributions([]),
        domain=ProductDomain(name="default", axes=[]),
        functions=Functions([]),
        progress=False,
        mode="FAST_COMPILE",
        likelihood=likelihood,
    )
    lp = model.log_prob
    assert isinstance(lp, TensorConstant)
    fn = pytensor.function([], lp)
    assert float(fn()) == pytest.approx(0.0)


def test_log_prob_aux_distributions_contributes_to_log_prob():
    """aux_distributions present → those distribution values enter log_prob."""
    ws = Workspace(**_WS_AUX)
    model = ws.model(ws.analyses["A"], progress=False)
    lp_without_aux = model.log_prob  # would be different without "constraint"

    # Build a workspace identical except aux_distributions removed.
    ws_no_aux = Workspace(
        **{
            **_WS_AUX,
            "likelihoods": [
                {"name": "L", "distributions": ["gauss1"], "data": ["data1"]}
            ],
        }
    )
    model_no_aux = ws_no_aux.model(ws_no_aux.analyses["A"], progress=False)
    lp_without_aux_expr = model_no_aux.log_prob

    inputs_with = {
        v.name: v for v in explicit_graph_inputs([lp_without_aux]) if v.name is not None
    }
    inputs_without = {
        v.name: v
        for v in explicit_graph_inputs([lp_without_aux_expr])
        if v.name is not None
    }
    fn_with = pytensor.function(list(inputs_with.values()), lp_without_aux)
    fn_without = pytensor.function(list(inputs_without.values()), lp_without_aux_expr)

    common = {"x_obs": np.array([1.0, 2.0, 3.0]), "mean": np.float64(0.0)}
    val_with = float(fn_with(**common, alpha=np.float64(0.0)))
    val_without = float(fn_without(**common))

    # With the constraint, log_prob gains an extra log(N(0|0,1)) term.
    assert val_with != pytest.approx(val_without)
    assert val_with < val_without  # constraint N(0|0,1) < 1, so log < 0


def test_log_prob_aux_unknown_distribution_is_silently_skipped():
    """aux_distributions names not in model.distributions are skipped without error."""
    ws = Workspace(
        **{
            **_WS_AUX,
            "likelihoods": [
                {
                    "name": "L",
                    "distributions": ["gauss1"],
                    "data": ["data1"],
                    # "nonexistent" is not a distribution in the workspace.
                    "aux_distributions": ["constraint", "nonexistent"],
                }
            ],
        }
    )
    model = ws.model(ws.analyses["A"], progress=False)
    # Should not raise; "nonexistent" is simply skipped.
    lp = model.log_prob
    assert hasattr(lp, "type")
